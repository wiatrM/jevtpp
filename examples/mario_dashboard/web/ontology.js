// Domain renderer for the versioned jevt.knowledge_view.v1 evidence graph.
// A model can emphasize an existing fact; it never creates a verified edge.
(() => {
  const get=id=>document.getElementById(id);
  const group={place:0,agent:1,powerup:1,action:1,skill:1,state:2,enemy:2,
    action_variant:2,enemy_type:3,danger_zone:3,motion:4,effect:4,outcome:4};
  const palette={observed:'#3dff74',empirical:'#69d7e6',confirmed:'#b9ff8f',
    uncertain:'#63836c',unresolved:'#63836c',confirmed_eliminated:'#b9ff8f'};
  let selected=null,regions=[],last=null,lastPayload=null;
  const nodeColor=n=>n.model_focus?'#f0ad27':palette[n.status]||'#6eaa81';
  const short=s=>String(s||'').length>23?String(s).slice(0,20)+'…':String(s||'');
  function inspect(view){
    const node=view.nodes.find(n=>n.id===selected);
    get('ontologyInspectTitle').textContent=node?node.label:'CLICK A FACT OR USE ARROW KEYS';
    get('ontologyInspect').textContent=node?JSON.stringify({fact:node,
      relations:view.edges.filter(e=>e.from===node.id||e.to===node.id),
      distinction:'observed / empirical / confirmed / uncertain'},null,2):
      'Select a node to inspect its provenance, samples and linked relations.';
  }
  function render(payload){
    const view=payload.ontology||{nodes:[],edges:[],model_focus:[]};
    last=view;lastPayload=payload;
    get('ontologyScope').textContent=view.nodes.length?
      `${view.scope==='cross_run_session'?'RETAINED':'RUN'} · ${view.runs_observed||1} RUNS · FRAME ${view.frame??'?'} · MEMORY GEN ${view.generation??'?'}`:
      'Run scoped · resets at GAME OVER';
    get('ontologyFocus').textContent=`VERIFIED MODEL FOCUS: ${view.model_focus?.join(', ')||'none (stale, deferred or no candidate)'}`;
    get('ontologyInfluence').textContent=`KNOWLEDGE ACTIONS ${payload.knowledge_actions_run||0} THIS RUN · ${payload.knowledge_actions_session||0} SESSION`;
    get('ontologyJson').textContent=JSON.stringify(view,null,2);
    get('ontologyModelInput').textContent=payload.model_knowledge_input?
      JSON.stringify(payload.model_knowledge_input,null,2):'No ranked plan request yet.';
    const canvas=get('ontologyCanvas'),w=canvas.clientWidth||850,h=390,d=window.devicePixelRatio||1;
    if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){
      canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);
    }
    const c=canvas.getContext('2d');c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,w,h);
    c.fillStyle='#09170e';c.fillRect(0,0,w,h);
    regions=[];
    if(!view.nodes.length){
      get('ontologyCount').textContent='0 FACTS · 0 RELATIONS';
      c.fillStyle='#63836c';c.font='12px monospace';
      c.fillText('Knowledge appears after RAM observations and verified transitions.',24,36);
      inspect(view);return;
    }
    const filter=get('ontologyFilter').value;
    const feedback=new Set(view.edges.filter(e=>
      ['observed_failure_context','observed_life_loss_near','nearby_at_life_loss'].includes(e.relation))
      .flatMap(e=>[e.from,e.to]));
    const candidates=view.nodes.filter(n=>filter==='all' ||
      (filter==='focus'&&n.model_focus) ||
      (filter==='feedback'&&feedback.has(n.id)) ||
      (filter==='empirical'&&['empirical','confirmed'].includes(n.status)));
    const sorted=candidates.slice().sort((a,b)=>
      Number(b.model_focus)-Number(a.model_focus)||
      Number(feedback.has(b.id))-Number(feedback.has(a.id))||
      Number(b.salience)-Number(a.salience)||
      Number(b.last_frame)-Number(a.last_frame));
    const selectedNode=view.nodes.find(n=>n.id===selected);
    const limits={enemy:3,action_variant:3,enemy_type:3,motion:2,effect:2,
      outcome:3,skill:3,danger_zone:4};
    const perKind={};
    const visible=[];
    for(const node of sorted){
      const count=perKind[node.kind]||0;
      if(count>=(limits[node.kind]||2))continue;
      visible.push(node);perKind[node.kind]=count+1;
      if(visible.length===36)break;
    }
    if(selectedNode&&!visible.some(n=>n.id===selected)&&filter==='all')visible.push(selectedNode);
    get('ontologyCount').textContent=`${view.nodes.length} FACTS · ${view.edges.length} RELATIONS · ${visible.length} SHOWN`;
    const byGroup=[[],[],[],[],[]];
    for(const node of visible)byGroup[group[node.kind]??2].push(node);
    const boxes={};
    const step=Math.max(148,(w-32)/5),boxW=Math.min(170,step-13);
    for(let col=0;col<5;col++){
      const items=byGroup[col];
      items.forEach((node,index)=>{
        const x=16+col*step,y=24+index*Math.min(57,340/Math.max(1,items.length));
        boxes[node.id]={x,y,w:boxW,h:42,node};
      });
    }
    for(const edge of view.edges){
      const a=boxes[edge.from],b=boxes[edge.to];if(!a||!b)continue;
      const picked=edge.from===selected||edge.to===selected;
      c.strokeStyle=picked?'#f0ad27':edge.status==='uncertain'?'#355241':'#295f43';
      c.lineWidth=picked?2.3:edge.salience>=.8?1.6:1;
      c.beginPath();c.moveTo(a.x+a.w,a.y+21);
      c.bezierCurveTo((a.x+a.w+b.x)/2,a.y+21,
        (a.x+a.w+b.x)/2,b.y+21,b.x,b.y+21);c.stroke();
      if(picked){
        c.fillStyle='#f0ad27';c.font='8px monospace';
        c.fillText(short(edge.relation),Math.min(w-150,(a.x+a.w+b.x)/2),((a.y+b.y)/2)+16);
      }
    }
    for(const {x,y,w:bw,h:bh,node} of Object.values(boxes)){
      c.fillStyle='#0a1c10';c.fillRect(x,y,bw,bh);
      c.strokeStyle=nodeColor(node);c.lineWidth=node.id===selected?2.5:node.model_focus?2:1;
      c.strokeRect(x+.5,y+.5,bw-1,bh-1);
      c.fillStyle=nodeColor(node);c.font='9px monospace';
      c.fillText(short(node.kind.toUpperCase()),x+7,y+13);
      c.fillStyle='#c9f8d3';c.font='11px monospace';
      c.fillText(short(node.label),x+7,y+30);
      regions.push({x,y,w:bw,h:bh,id:node.id});
    }
    inspect(view);
  }
  get('ontologyCanvas').addEventListener('click',event=>{
    const rect=event.currentTarget.getBoundingClientRect();
    const x=event.clientX-rect.left,y=event.clientY-rect.top;
    const hit=regions.find(r=>x>=r.x&&x<=r.x+r.w&&y>=r.y&&y<=r.y+r.h);
    if(hit){selected=hit.id;if(lastPayload)render(lastPayload);}
  });
  get('ontologyCanvas').addEventListener('keydown',event=>{
    if(!['ArrowRight','ArrowLeft','Enter'].includes(event.key)||!regions.length)return;
    event.preventDefault();
    const ids=regions.map(r=>r.id),at=ids.indexOf(selected);
    selected=ids[(at+(event.key==='ArrowLeft'?-1:1)+ids.length)%ids.length];
    if(lastPayload)render(lastPayload);
  });
  get('ontologyFilter').addEventListener('change',()=>{if(lastPayload)render(lastPayload);});
  window.renderOntology=render;
})();
