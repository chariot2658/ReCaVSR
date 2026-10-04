"use strict";
(() => {
  const catalogue = window.RECAVSR;
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const titles = {
    "real-real-world-021": "After dark, in the city",
    "real-real-world-009": "A portrait in motion",
    "real-videolq-001": "Crossing the city",
    "real-videolq-002": "Along the avenue",
    "real-videolq-008": "At the market",
    "real-videolq-024": "Across the waterfront",
    "real-videolq-035": "A walk down the street",
    "aigc-aigc-014": "An evening walk",
    "synthetic-udm10-002": "Under an open sky",
    "synthetic-youhq-000": "A moment in the crowd",
    "synthetic-youhq-004": "The old town square",
    "synthetic-youhq-006": "Around the corner",
    "synthetic-youhq-021": "Life on the reef",
    "streaming-real-world-002": "Through the green",
    "streaming-aigc-001": "A generated world",
    "streaming-aigc-021": "A continuous portrait"
  };
  const categoryNames = {real:"Real-world",aigc:"AIGC",synthetic:"Synthetic",streaming:"Long-form"};
  const clips = catalogue.clips.map(c => ({...c,title:titles[c.id] || c.title}));
  const byId = Object.fromEntries(clips.map(c => [c.id,c]));
  const label = m => catalogue.methods[m] || m;
  const fmt = sec => `${Math.floor(sec/60).toString().padStart(2,"0")}:${Math.floor(sec%60).toString().padStart(2,"0")}`;
  const meta = clip => `${clip.width.toLocaleString()} × ${clip.height.toLocaleString()} · ${clip.frames.toLocaleString()} frames · 4× upscaling`;
  const players = new Set();
  let showComparison = () => {};
  const adjacentClip = (ids, current, delta) => ids[(ids.indexOf(current) + delta + ids.length) % ids.length];
  const make = (tag, attrs={}, text="") => {
    const el=document.createElement(tag);
    for(const [k,v] of Object.entries(attrs)) el.setAttribute(k,String(v));
    el.textContent=text; return el;
  };

  class Player {
    constructor(root, clip, options={}) {
      this.root=root; this.clip=clip; this.options=options;
      this.mode="compare";
      this.baseline=options.baseline || "input";
      this.videoMap=new Map(); this.time=0; this.intent=false; this.playing=false;
      this.starting=false; this.buffering=false; this.disposed=false; this.request=0;
      this.region=[.62,.28]; this.zoom=3; this.autoStarted=false;
      this.render(); players.add(this);
      this.observer=new IntersectionObserver(entries=>{
        for(const entry of entries) {
          if(entry.isIntersecting) {
            this.ensure();
            if(this.options.autoplay && !this.autoStarted && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
              this.autoStarted=true; this.play();
            }
          } else this.pause();
        }
      },{threshold:.15});
      this.observer.observe(root);
    }
    render() {
      const c=this.clip;
      this.root.style.setProperty('--video-ratio',c.width/c.height);
      this.root.innerHTML=`<div class="player-stage${this.options.detail?' detail-layout':''}">
        <div class="screen mode-${this.mode}"><div class="layer layer-ours"><img class="poster" src="${c.poster}" alt="${c.title} — ReCaVSR output"></div><div class="layer layer-left"></div><span class="video-label left"></span><span class="video-label ours">RECAVSR</span><div class="divider"><input class="wipe-range" aria-label="Input or baseline comparison divider" type="range" min="0" max="100" value="50"></div>${this.options.detail?'<div class="crop-region"></div>':''}</div>
        ${this.options.detail?'<div class="loupe"><span class="video-label">RECAVSR / 3× DETAIL</span></div>':''}
        </div><div class="player-controls"><div class="transport"><button class="play-toggle" aria-label="Play video">▶</button><input class="seek-range" type="range" aria-label="Video frame" min="1" max="${c.frames}" value="1" step="1"><output class="time-label"></output></div><div class="player-modes" role="group" aria-label="Video view"></div><button class="full-button" aria-label="Expand video">⛶</button></div><div class="player-status" role="status" aria-live="polite"></div>`;
      this.screen=$('.screen',this.root); this.status=$('.player-status',this.root);
      this.playButton=$('.play-toggle',this.root); this.range=$('.seek-range',this.root);
      if(this.options.onNavigate) {
        for(const [direction,delta,symbol] of [['previous',-1,'‹'],['next',1,'›']]) {
          const name=direction==='previous'?'Previous video':'Next video';
          const button=make('button',{class:`clip-nav ${direction}`,type:'button','aria-label':name,title:name},symbol);
          button.disabled=this.options.navigationCount<2;
          button.onclick=event=>{
            event.stopPropagation();this.options.onNavigate(delta);
            $(`.clip-nav.${direction}`,this.root)?.focus({preventScroll:true});
          };
          this.screen.append(button);
        }
      }
      $('.player-modes',this.root).remove();
      $('.video-label.left',this.root).textContent=label(this.baseline);
      this.playButton.onclick=()=>this.intent?this.pause():this.play();
      this.range.oninput=()=>this.seekFrame(Number(this.range.value));
      $('.wipe-range',this.root).oninput=e=>this.screen.style.setProperty('--split',e.target.value+'%');
      $('.full-button',this.root).onclick=async()=>{
        try {
          if(document.fullscreenElement) await document.exitFullscreen();
          else await $('.player-stage',this.root).requestFullscreen();
        } catch { this.setStatus('Fullscreen is not available in this browser.'); }
      };
      if(this.options.detail) {
        // The full-frame drag gesture belongs to the comparison divider.
        // Detail regions are selected with the adjacent preset controls.
        this.resizeObserver=new ResizeObserver(()=>this.updateCrop()); this.resizeObserver.observe(this.screen);
      }
      this.updateTime();
    }
    setStatus(text,error=false) { this.status.textContent=text; this.status.classList.toggle('error',error); }
    video(key,method) {
      if(this.videoMap.has(key)) return this.videoMap.get(key);
      const source=this.clip.sources[method];
      const v=make('video',{playsinline:'',preload:'metadata','aria-label':`${label(method)} — ${this.clip.title}`});
      v.muted=true; v.dataset.method=method; v.dataset.fps=source.fps;
      v.src=source.url;
      v.addEventListener('loadeddata',()=>{
        if(this.disposed) return;
        if(key==='ours') $('.poster',this.root)?.setAttribute('hidden','');
        if(this.activeVideos().every(x=>x.readyState>=2) && !this.intent) this.setStatus('');
        this.updateCrop();
      });
      v.addEventListener('error',()=>{
        if(this.disposed) return;
        this.pause(); this.setStatus(`Unable to load ${label(method)}.`,true);
        const retry=make('button',{},'Retry'); retry.onclick=()=>{v.load(); this.setStatus('Loading video…');}; this.status.append(retry);
      });
      v.addEventListener('waiting',()=>{
        if(!this.playing || !this.intent || this.disposed) return;
        this.buffering=true; this.playing=false; cancelAnimationFrame(this.raf);
        this.activeVideos().forEach(x=>x.pause()); this.setStatus('Buffering the same frame across both views…');
      });
      v.addEventListener('canplay',()=>{
        if(this.buffering && this.intent && this.activeVideos().every(x=>x.readyState>=3)) {
          this.buffering=false; this.play();
        }
      });
      v.addEventListener('ended',()=>{
        if(this.intent && v===this.master()) { this.time=0; this.playing=false; this.buffering=false; this.align(); this.play(); }
      });
      this.videoMap.set(key,v);
      (key==='crop'?$('.loupe',this.root):key==='left'?$('.layer-left',this.root):$('.layer-ours',this.root)).append(v);
      return v;
    }
    ensure() {
      if(this.disposed) return;
      this.video('ours','recavsr');
      if(this.mode!=='ours') this.video('left',this.baseline);
      if(this.options.detail) this.video('crop','recavsr');
      if(this.activeVideos().some(v=>v.readyState<1)) this.setStatus('Loading video…');
    }
    activeVideos() {
      const keys=this.mode==='input'?['left']:this.mode==='ours'?['ours']:['ours','left'];
      if(this.options.detail) keys.push('crop');
      return keys.map(k=>this.videoMap.get(k)).filter(Boolean);
    }
    master() { return this.videoMap.get(this.mode==='input'?'left':'ours'); }
    waitForMetadata(v) {
      if(v.readyState>=1) return Promise.resolve();
      return new Promise((resolve,reject)=>{
        const done=()=>{clearTimeout(timer);v.removeEventListener('loadedmetadata',success);v.removeEventListener('error',failure);};
        const success=()=>{done();resolve();}; const failure=()=>{done();reject(new Error('Video unavailable'));};
        const timer=setTimeout(failure,25000); v.addEventListener('loadedmetadata',success,{once:true});v.addEventListener('error',failure,{once:true});
      });
    }
    fps(v) { return Number(v.dataset.fps)||this.clip.fps; }
    align() {
      for(const v of this.activeVideos()) {
        const t=this.time*this.clip.fps/this.fps(v);
        if(v.readyState>=1 && Math.abs(v.currentTime-t)>.012) v.currentTime=Math.min(t,Math.max(0,v.duration-.001));
        v.playbackRate=this.clip.fps/this.fps(v);
      }
    }
    async play() {
      if(this.starting || this.playing || this.disposed) return;
      players.forEach(p=>{if(p!==this)p.pause();});
      this.ensure(); this.intent=true; this.starting=true; this.buffering=false;
      const request=++this.request; this.playButton.textContent='Ⅱ';this.playButton.setAttribute('aria-label','Pause video');
      this.setStatus('Preparing playback…');
      try {
        const videos=this.activeVideos(); videos.forEach(v=>v.preload='auto');
        await Promise.all(videos.map(v=>this.waitForMetadata(v)));
        if(this.disposed || request!==this.request || !this.intent) return;
        if(this.time>=(this.clip.frames-1)/this.clip.fps) this.time=0;
        this.align();
        await Promise.all(videos.map(v=>v.play()));
        if(this.disposed || request!==this.request || !this.intent) {videos.forEach(v=>v.pause());return;}
        this.playing=true;this.starting=false;this.setStatus('');this.tick();
      } catch {
        if(this.disposed || request!==this.request) return;
        this.pause();this.setStatus('Playback could not start. Press play to retry.',true);
      }
    }
    pause() {
      this.intent=false;this.playing=false;this.starting=false;this.buffering=false;this.request++;
      cancelAnimationFrame(this.raf);this.videoMap.forEach(v=>v.pause());
      this.playButton.textContent='▶';this.playButton.setAttribute('aria-label','Play video');
      if(!this.status.classList.contains('error')) this.setStatus('');
    }
    tick() {
      if(!this.playing || this.disposed) return;
      const master=this.master();this.time=master.currentTime*this.fps(master)/this.clip.fps;
      for(const v of this.activeVideos()) {
        const target=this.time*this.clip.fps/this.fps(v);
        if(v!==master && !v.seeking && Math.abs(v.currentTime-target)>.07) v.currentTime=Math.min(target,v.duration-.001);
      }
      this.updateTime();this.raf=requestAnimationFrame(()=>this.tick());
    }
    updateTime() {
      const frame=Math.min(this.clip.frames,Math.floor(this.time*this.clip.fps+.001)+1);
      this.range.value=frame;
      $('.time-label',this.root).textContent=this.clip.category==='streaming'?`${frame.toLocaleString()} / ${this.clip.frames.toLocaleString()} f`:`${fmt(this.time)} / ${fmt(this.clip.frames/this.clip.fps)}`;
    }
    async seekFrame(frame) {
      this.pause();this.ensure();this.time=(Math.max(1,Math.min(frame,this.clip.frames))-1)/this.clip.fps;this.updateTime();
      const request=++this.request;
      try {await Promise.all(this.activeVideos().map(v=>this.waitForMetadata(v)));if(this.disposed||request!==this.request)return;this.align();}
      catch {if(!this.disposed)this.setStatus('Could not load this frame. Try again.',true);}
    }
    async setMode(mode) {
      const resume=this.intent;this.pause();this.mode=mode;
      this.screen.className='screen mode-'+mode;
      $$('.player-modes button',this.root).forEach(b=>b.setAttribute('aria-pressed',b.dataset.mode===mode));
      this.ensure();await this.seekFrame(Math.floor(this.time*this.clip.fps)+1);
      if(resume&&!this.disposed)this.play();
    }
    updateCrop() {
      if(!this.options.detail || this.disposed) return;
      const loupe=$('.loupe',this.root),video=this.videoMap.get('crop');
      const w=this.screen.clientWidth,h=this.screen.clientHeight,cw=loupe.clientWidth,ch=loupe.clientHeight;
      if(!w||!h||!cw||!ch)return;
      const rw=cw/(w*this.zoom),rh=ch/(h*this.zoom);
      const x=Math.max(rw/2,Math.min(1-rw/2,this.region[0])),y=Math.max(rh/2,Math.min(1-rh/2,this.region[1]));
      if(video) Object.assign(video.style,{width:w*this.zoom+'px',height:h*this.zoom+'px',left:(cw/2-x*w*this.zoom)+'px',top:(ch/2-y*h*this.zoom)+'px'});
      Object.assign($('.crop-region',this.root).style,{width:rw*100+'%',height:rh*100+'%',left:(x-rw/2)*100+'%',top:(y-rh/2)*100+'%'});
    }
    dispose() {
      this.pause();this.disposed=true;this.observer.disconnect();this.resizeObserver?.disconnect();
      this.videoMap.forEach(v=>{v.removeAttribute('src');v.load();});players.delete(this);
    }
  }
  document.addEventListener('visibilitychange',()=>{if(document.hidden)players.forEach(p=>p.pause());});

  if($('#hero-player')) {
    const featured=['real-real-world-021','synthetic-youhq-021','synthetic-youhq-006'];
    let hero;
    function selectFeatured(id) {
      hero?.dispose();
      hero=new Player($('#hero-player'),byId[id],{autoplay:true,navigationCount:featured.length,
        onNavigate:delta=>selectFeatured(adjacentClip(featured,hero.clip.id,delta))});
      $$('#hero-choices button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.clip===id));
    }
    selectFeatured(featured[0]);
    featured.forEach((id,i)=>{
      const c=byId[id],b=make('button',{class:'hero-choice','data-clip':id,'aria-pressed':i===0,'aria-label':c.title,title:c.title});
      b.append(make('img',{src:c.poster,alt:''}));b.onclick=()=>selectFeatured(id);$('#hero-choices').append(b);
    });
    const selected=['synthetic-youhq-006','real-videolq-001','aigc-aigc-014','real-real-world-009','real-videolq-024','synthetic-youhq-021'];
    let category='all',expanded=false,dialogPlayer=null,galleryClips=[];
    const dialog=$('#video-dialog');
    const closeDialog=()=>{dialogPlayer?.dispose();dialogPlayer=null;$('#dialog-player').replaceChildren();};
    dialog.addEventListener('close',closeDialog);$('.dialog-close',dialog).onclick=()=>dialog.close();
    dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
    function openClip(clip) {
      const autoplay=dialog.open?!!dialogPlayer?.intent:true;
      dialogPlayer?.dispose();
      players.forEach(p=>p.pause());$('#dialog-title').textContent=clip.title;
      $('#dialog-category').textContent=categoryNames[clip.category]+' / SELECTED RESULT';
      $('#dialog-meta').textContent=meta(clip);
      $('#dialog-compare').onclick=event=>{
        event.preventDefault();dialog.close();showComparison(clip.id);
        location.hash='comparison';
        $('#comparison').scrollIntoView({behavior:'smooth',block:'start'});
      };
      if(!dialog.open)dialog.showModal();
      dialogPlayer=new Player($('#dialog-player'),clip,{autoplay,navigationCount:galleryClips.length,
        onNavigate:delta=>openClip(byId[adjacentClip(galleryClips.map(c=>c.id),clip.id,delta)])});
    }
    function renderGallery() {
      const group=category==='all'&&!expanded?selected.map(id=>byId[id]):clips.filter(c=>c.category!=='streaming'&&(category==='all'||c.category===category));
      galleryClips=group;
      $('#gallery').replaceChildren();
      group.forEach(c=>{
        const b=make('button',{class:'gallery-card','aria-label':'Watch '+c.title});
        b.innerHTML=`<div class="card-image"><img loading="lazy" src="${c.poster}" alt="${c.title}"><span class="card-category">${categoryNames[c.category]}</span><span class="card-open" aria-hidden="true">↗</span></div><div class="card-caption"><h3>${c.title}</h3><span>${c.width.toLocaleString()} × ${c.height.toLocaleString()}</span></div>`;
        b.onclick=()=>openClip(c);$('#gallery').append(b);
      });
      $('#gallery-count').textContent=String(group.length).padStart(2,'0')+' SELECTED SCENES';
      $('#all-clips').hidden=category!=='all';$('#all-clips').innerHTML=expanded?'Show selected videos <span>↑</span>':'View all 13 short videos <span>↗</span>';
    }
    $$('.tabs button').forEach(b=>b.onclick=()=>{
      category=b.dataset.category;expanded=false;$$('.tabs button').forEach(x=>x.setAttribute('aria-pressed',x===b));renderGallery();
    });
    $('#all-clips').onclick=()=>{expanded=!expanded;renderGallery();};renderGallery();
    const longs=['streaming-real-world-002','streaming-aigc-001','streaming-aigc-021'];
    let long;
    function selectLong(id) {
      const autoplay=!!long?.intent;
      long?.dispose();long=new Player($('#long-player'),byId[id],{autoplay,navigationCount:longs.length,
        onNavigate:delta=>selectLong(adjacentClip(longs,long.clip.id,delta))});
      $('#long-original').href=long.clip.sources.recavsr.url;
      $$('#long-choices button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.clip===id));
    }
    selectLong(longs[0]);
    longs.forEach((id,i)=>{
      const c=byId[id],b=make('button',{class:'long-choice','data-clip':id,'aria-pressed':i===0});
      b.innerHTML=`<img src="${c.poster}" alt=""><span><strong>${c.title}</strong><small>${c.group.endsWith('aigc')?'AIGC':'Real-world'} · 1,000 frames</small></span>`;
      b.onclick=()=>selectLong(id);
      $('#long-choices').append(b);
    });
  }
  if($('#compare-player')) {
    const collection=$('#collection'),scene=$('#scene'),baseline=$('#baseline');
    const requested=new URLSearchParams(location.search).get('clip');
    let active=byId[requested]||byId['real-real-world-021'],player;
    let initialized=false;
    function updatePlayer() {
      const autoplay=!!player?.intent;
      player?.dispose();player=new Player($('#compare-player'),active,{comparison:true,baseline:baseline.value,autoplay,
        navigationCount:scene.options.length,
        onNavigate:delta=>selectScene(adjacentClip([...scene.options].map(o=>o.value),active.id,delta))});
      $('#comparison-meta').textContent=meta(active)+' · '+active.group.split('/')[1].replace('videolq','VideoLQ').replace('youhq','YouHQ').replace('udm10','UDM10')+' · Sample '+active.sample;
      if(initialized || requested) {
        const url=new URL(location.href);url.searchParams.set('clip',active.id);
        history.replaceState(null,'',url.pathname+url.search+url.hash);
      }
      initialized=true;
    }
    function selectScene(id) {
      active=byId[id];scene.value=id;const previous=baseline.value;
      baseline.replaceChildren();
      const order=['flashvsr-tiny','seedvr2','dove','swiftvr','realviformer','realbasicvsr','sparkvsr','mgld-vsr','dloral','star','input'];
      order.filter(m=>active.sources[m]).forEach(m=>baseline.append(make('option',{value:m},label(m))));
      if(active.sources[previous]&&previous!=='recavsr')baseline.value=previous;
      $$('#scene-strip button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.clip===id));updatePlayer();
    }
    function updateCollection() {
      const group=clips.filter(c=>collection.value==='all'||c.category===collection.value);
      scene.replaceChildren();$('#scene-strip').replaceChildren();
      group.forEach(c=>{
        scene.append(make('option',{value:c.id},c.title));
        const b=make('button',{class:'scene-tile','data-clip':c.id,'aria-label':c.title});
        b.innerHTML=`<img src="${c.poster}" alt=""><span>${categoryNames[c.category]} / ${c.sample}</span>`;
        b.onclick=()=>selectScene(c.id);$('#scene-strip').append(b);
      });
      selectScene(group.some(c=>c.id===active.id)?active.id:group[0].id);
    }
    showComparison=id=>{
      if(!byId[id])return;
      active=byId[id];collection.value='all';updateCollection();
    };
    collection.onchange=updateCollection;scene.onchange=()=>selectScene(scene.value);baseline.onchange=updatePlayer;updateCollection();
  }
})();
