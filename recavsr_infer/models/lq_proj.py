from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


@dataclass
class CausalConv3dStreamingState:
    """Minimal temporal history needed by one strided causal convolution."""

    history: torch.Tensor | None = None
    seen_frames: int = 0


@dataclass
class LQ4xProjStreamingState:
    """Two-level causal-convolution state for incremental LQ projection."""

    conv1: CausalConv3dStreamingState
    conv2: CausalConv3dStreamingState


class ChannelRMSNorm3d(nn.Module):
    """RMS-normalize a video feature map over its channel dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=1, keepdim=True) + self.eps
        )
        return (normalized * self.weight).to(dtype=dtype)


class CausalConv3d(nn.Conv3d):
    """A Conv3d whose temporal receptive field never includes future frames."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temporal_stride: int,
        spatial_stride: int,
    ) -> None:
        kernel_size = (4, 3, 3)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
        )
        self.temporal_padding = kernel_size[0] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad order for a 5-D tensor is W, H, T. Replicating the first
        # frame follows Wan's causal video convention without future leakage.
        x = F.pad(
            x,
            (0, 0, 0, 0, self.temporal_padding, 0),
            mode="replicate",
        )
        return super().forward(x)

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: CausalConv3dStreamingState,
    ) -> torch.Tensor:
        """Project a consecutive input chunk without resetting stride alignment."""
        if x.shape[2] <= 0:
            raise ValueError("A streaming convolution chunk cannot be empty.")
        kernel_frames = int(self.kernel_size[0])
        stride_frames = int(self.stride[0])
        seen_frames = state.seen_frames
        next_endpoint = (
            (seen_frames + stride_frames - 1) // stride_frames
        ) * stride_frames
        last_input_index = seen_frames + x.shape[2] - 1
        if next_endpoint > last_input_index:
            raise ValueError(
                "Streaming LQ chunk is too short to emit the next strided "
                f"convolution output: seen={seen_frames}, chunk={x.shape[2]}."
            )
        last_endpoint = (
            last_input_index // stride_frames
        ) * stride_frames

        history = state.history
        if history is None:
            combined = x
            combined_start = seen_frames
        else:
            if history.shape[:2] != x.shape[:2] or history.shape[3:] != x.shape[3:]:
                raise ValueError("Streaming convolution history shape changed.")
            combined = torch.cat((history, x), dim=2)
            combined_start = seen_frames - history.shape[2]

        needed_start = next_endpoint - (kernel_frames - 1)
        actual_start = max(needed_start, 0)
        local_start = actual_start - combined_start
        local_end = last_endpoint - combined_start + 1
        convolution_input = combined[:, :, local_start:local_end]
        if needed_start < 0:
            left_padding = -needed_start
            first_frame = combined[:, :, :1].expand(-1, -1, left_padding, -1, -1)
            convolution_input = torch.cat((first_frame, convolution_input), dim=2)

        output = F.conv3d(
            convolution_input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        expected_outputs = 1 + (last_endpoint - next_endpoint) // stride_frames
        if output.shape[2] != expected_outputs:
            raise RuntimeError(
                "Streaming causal convolution emitted an unexpected frame count: "
                f"expected={expected_outputs}, got={output.shape[2]}."
            )

        # Own only the required history. A detached slice still retains the
        # entire chunk's storage, unnecessarily increasing persistent memory.
        state.history = combined[:, :, -(kernel_frames - 1) :].detach().clone()
        state.seen_frames += int(x.shape[2])
        return output


class LQ4xProj(nn.Module):
    """Project upsampled RGB LQ video into Wan2.2 TI2V-5B patch tokens.

    Two causal temporal convolutions each downsample by two, producing
    ``ceil(T / 4) == 1 + (T - 1) // 4`` output frames. Spatially, a 16x16
    pixel unshuffle followed by a stride-2 convolution produces one token per
    32x32 RGB input region, aligned with Wan2.2's VAE and DiT patchification.

    The final projection is zero-initialized so adding this condition (or an
    additive previous-SR condition alongside it) does not perturb the
    pretrained transformer before optimization starts.
    """

    temporal_downsample_factor = 4
    spatial_downsample_factor = 32
    pixel_unshuffle_factor = 16

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3072,
        hidden_dim1: int = 2048,
        hidden_dim2: int = 3072,
    ) -> None:
        super().__init__()
        for name, value in {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "hidden_dim1": hidden_dim1,
            "hidden_dim2": hidden_dim2,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        unshuffled_channels = (
            in_dim * self.pixel_unshuffle_factor * self.pixel_unshuffle_factor
        )
        self.conv1 = CausalConv3d(
            unshuffled_channels,
            hidden_dim1,
            temporal_stride=2,
            spatial_stride=1,
        )
        self.norm1 = ChannelRMSNorm3d(hidden_dim1)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(
            hidden_dim1,
            hidden_dim2,
            temporal_stride=2,
            spatial_stride=2,
        )
        self.norm2 = ChannelRMSNorm3d(hidden_dim2)
        self.act2 = nn.SiLU()

        self.out_proj = nn.Linear(hidden_dim2, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in (self.conv1, self.conv2):
            nn.init.normal_(conv.weight, mean=0.0, std=0.02)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def create_streaming_state(self) -> LQ4xProjStreamingState:
        return LQ4xProjStreamingState(
            conv1=CausalConv3dStreamingState(),
            conv2=CausalConv3dStreamingState(),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        _, _, frames, height, width = self._validate_video(video)
        x = self._pixel_unshuffle(video)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))

        expected_frames = 1 + (frames - 1) // self.temporal_downsample_factor
        return self._project_output(
            x,
            expected_frames=expected_frames,
            expected_height=height // self.spatial_downsample_factor,
            expected_width=width // self.spatial_downsample_factor,
        )

    def forward_streaming(
        self,
        video: torch.Tensor,
        state: LQ4xProjStreamingState,
    ) -> torch.Tensor:
        """Project a consecutive RGB chunk exactly as full-sequence forward."""
        _, _, _, height, width = self._validate_video(video)
        x = self._pixel_unshuffle(video)
        x = self.act1(self.norm1(self.conv1.forward_streaming(x, state.conv1)))
        x = self.act2(self.norm2(self.conv2.forward_streaming(x, state.conv2)))
        return self._project_output(
            x,
            expected_frames=int(x.shape[2]),
            expected_height=height // self.spatial_downsample_factor,
            expected_width=width // self.spatial_downsample_factor,
        )

    def _validate_video(
        self,
        video: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if video.ndim != 5:
            raise ValueError(
                "video must have shape [B, C, T, H, W], "
                f"got {tuple(video.shape)}."
            )
        shape = tuple(int(value) for value in video.shape)
        _, channels, frames, height, width = shape
        expected_channels = self.conv1.in_channels // (
            self.pixel_unshuffle_factor**2
        )
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels, got {channels}."
            )
        if frames < 1:
            raise ValueError("video must contain at least one frame.")
        if (
            height % self.spatial_downsample_factor != 0
            or width % self.spatial_downsample_factor != 0
        ):
            raise ValueError(
                "Input height and width must be divisible by "
                f"{self.spatial_downsample_factor}, got {(height, width)}."
            )
        return shape

    def _pixel_unshuffle(self, video: torch.Tensor) -> torch.Tensor:
        return rearrange(
            video,
            "b c t (h ph) (w pw) -> b (c ph pw) t h w",
            ph=self.pixel_unshuffle_factor,
            pw=self.pixel_unshuffle_factor,
        )

    def _project_output(
        self,
        x: torch.Tensor,
        *,
        expected_frames: int,
        expected_height: int,
        expected_width: int,
    ) -> torch.Tensor:
        if x.shape[2:] != (expected_frames, expected_height, expected_width):
            raise RuntimeError(
                "LQ projection shape mismatch: expected "
                f"{(expected_frames, expected_height, expected_width)}, "
                f"got {tuple(x.shape[2:])}."
            )

        x = rearrange(x, "b c t h w -> b (t h w) c")
        return self.out_proj(x)
