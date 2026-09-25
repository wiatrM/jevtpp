const actions=['noop','right','right_jump','right_run','right_run_jump','jump','left'];
const $=id=>document.getElementById(id);
let paused=false,lastPayload=null;
let graphHitAreas=[],hoveredGraphNode=null,pinnedGraphNode='telemetry';

function number(value,digits=0){return Number(value??0).toFixed(digits)}
function title(value){return String(value??'—').replaceAll('_',' ').toUpperCase()}
function clamp(value){return Math.max(0,Math.min(1,Number(value)||0))}

function renderProbabilities(decision){
  if(decision?.judgment_kind==='plan'){
    const entries=[...Object.entries(decision.plan_scores||{}),['defer',decision.defer_score||0]];
    $('probabilities').replaceChildren(...entries.map(([id,score])=>{
      const row=document.createElement('div');row.className='prob-row'+(id===decision.model_plan||(id==='defer'&&decision.selected_slot===6)?' selected':'');
      const label=document.createElement('span');label.textContent=title(decision.candidate_labels?.[id]||id);label.title=id;
      const track=document.createElement('div');track.className='prob-track';const bar=document.createElement('i');bar.style.width=`${clamp(score)*100}%`;track.append(bar);
      const value=document.createElement('output');value.textContent=`${number(clamp(score)*100,1)}%`;row.append(label,track,value);return row;
    }));return;
  }
  const probabilities=decision?.goal_probabilities||{};
  const selected=decision?.model_goal||decision?.active_goal||'';
  const goals=['finish_fast','stomp_enemy','collect_powerup','collect_coins','score_attack','recover_momentum'];
  $('probabilities').innerHTML=goals.map(action=>{
    const value=clamp(probabilities[action]);
    return `<div class="prob-row ${action===selected?'selected':''}"><span>${title(action)}</span><div class="prob-track"><i style="width:${value*100}%"></i></div><output>${number(value*100,1)}%</output></div>`;
  }).join('');
}

function renderState(payload){
  const state=payload.state||{},player=state.player||{},terrain=state.terrain||{},hazard=state.hazard||{},episode=state.episode||{},recent=state.recent_control||{},detectors=state.detectors||{},optimization=state.optimization||{},session=payload.session||state.session||{},landing=payload.decision?.landing_plan;
  const trusted=terrain.last_grounded_preview||{};
  const facts=[['POSITION',`${player.x??'—'} / ${player.y??'—'}`],['VELOCITY',`${player.vx??0} / ${player.vy??0}`],['JUMP PHASE',title(player.jump_phase)],['GROUND',player.grounded?'CONTACT':'AIRBORNE'],['NEXT ENEMY',hazard.nearest_enemy_kind?`${title(hazard.nearest_enemy_kind)} · ${hazard.enemy_distance}px`:'NONE'],['CONTACT ETA',hazard.estimated_contact_frames==null?'—':`${hazard.estimated_contact_frames} FR`],['LIVE GAP',terrain.gap_distance<999?`${terrain.gap_distance}px · ${terrain.gap_width_tiles_visible}T`:'CLEAR / UNKNOWN'],['TRUSTED GAP',trusted.gap_distance_tiles==null?'CLEAR':`${trusted.gap_distance_tiles}T · WIDTH ${trusted.gap_width_tiles_visible||0}T`],['OBSTACLE',terrain.obstacle_distance<999?`${terrain.obstacle_distance}px · ${terrain.obstacle_height_tiles}T`:'CLEAR / UNKNOWN'],['TAKEOFF',hazard.takeoff_deadline_frames==null?'—':`${hazard.takeoff_deadline_frames} FR`],['LANDING TARGET',landing?`${landing.left_x}–${landing.right_x} · Y${landing.y}`:'NONE'],['OBSERVED PLATFORMS',state.collision?.landing_surfaces?.length??0],['CONTROL',`${title(recent.outcome)} · ${recent.frames_observed||0} FR`],['PACE',`${number(optimization.progress_pixels_per_second,1)} PX/S`]];
  $('stateFacts').innerHTML=facts.map(([key,value])=>`<div class="fact"><small>${key}</small><strong>${value}</strong></div>`).join('');
  const liveGrid=payload.debug?.local_grid||[],trustedGrid=payload.debug?.trusted_grid||[];
  const useTrusted=!String(terrain.observation_reliability).startsWith('high')&&trustedGrid.length;
  $('tileGrid').textContent=(useTrusted?trustedGrid:liveGrid).join('\n')||'RAM GRID UNAVAILABLE';
  $('tileGrid').dataset.source=useTrusted?'last trusted grounded sample':'live RAM sample';
  $('gridSource').textContent=useTrusted?'LAST TRUSTED GROUND':'LIVE RAM';
  const detectorFacts=[['HOSTILES',detectors.hostiles_visible??0],['POWER-UPS',detectors.powerups_visible??0],['TARGET POWER-UP',detectors.powerup_distance<999?`${detectors.powerup_distance}px`:'NONE'],['KILLS',detectors.eliminations_detected??0],['COINS',detectors.coins??0],['SCORE',optimization.score??0]];
  $('detectorGrid').innerHTML=detectorFacts.map(([key,value])=>`<div><small>${key}</small><strong>${value}</strong></div>`).join('');
  const threats=[];
  if(hazard.jump_must_start_this_decision) threats.push('takeoff deadline is inside this decision');
  if(hazard.contact_within_reaction_horizon) threats.push('enemy contact inside reaction horizon');
  if(terrain.gap_ahead) threats.push(`gap ${terrain.gap_distance_tiles} tiles ahead`);
  if(terrain.obstacle_ahead) threats.push(`obstacle ${terrain.obstacle_distance_tiles} tiles ahead`);
  $('situation').textContent=threats.length?threats.join(' · '):`clear forward path · ${player.grounded?'grounded':'trajectory in progress'}`;
  $('timeLeft').textContent=episode.time_left??'—'; $('lives').textContent=session.lives_remaining??episode.lives??'—'; $('score').textContent=optimization.score??0;
  const cleared=Math.min(32,Math.max(0,Number(session.stages_cleared)||0));
  $('worldStage').textContent=`${session.world??state.level?.world??'—'}-${session.stage??state.level?.stage??'—'}`;
  $('runId').textContent=session.run_id??'—'; $('stagesCleared').textContent=`${cleared} / 32`;
  $('deathCount').textContent=session.deaths??'—'; $('memoryGeneration').textContent=session.memory_generation??'—';
  $('stageProgressText').textContent=session.game_complete?`GAME COMPLETE · ${cleared} STAGES PLAYED`:`${cleared} / 32 STAGES · CURRENT X ${player.x??'—'} PX`;
  $('progressBar').style.width=`${session.game_complete?100:cleared/32*100}%`;
  const phase=payload.game_phase?.phase||payload.status||'unknown';
  $('sessionPhase').textContent=title(phase);
  $('sessionPhase').classList.toggle('terminal',Boolean(session.game_over||session.game_complete));
}

function renderSituation(payload){
  const decision=payload.decision||{},state=payload.state||{},timing=payload.timing||{},runtime=payload.runtime||{};
  const planMode=decision.judgment_kind==='plan';
  const weights=Object.entries((planMode?decision.plan_scores:decision.goal_probabilities)||{}).sort((a,b)=>Number(b[1])-Number(a[1]));
  const top=weights.slice(0,3).map(([goal,weight])=>`${title(goal)} ${number(clamp(weight)*100,1)}%`).join(' · ');
  $('gpuDevice').textContent=runtime.validated?`${runtime.device_name} · ${runtime.device}`:'DEVICE NOT VERIFIED';
  const baseModelId=runtime.base_model_id||decision.model_id;
  const isLaya=runtime.model_kind==='laya';
  $('modelStack').textContent=baseModelId?(isLaya?`LAYA · ${baseModelId} · CUDA · JEVT++ C++`:`OPEN-JEV HEAD + ${baseModelId} BASE · JEVT++ C++`):'VERIFYING MODEL ID';
  const resolvedGoal=decision.intent_accepted?title(decision.active_goal):'REFLEX / NO ACCEPTED GOAL';
  const session=payload.session||{},phase=payload.game_phase||{},delta=payload.learning_delta||{},risk=payload.risk_policy||{};
  $('situationSummary').textContent=`World ${session.world??'?' }-${session.stage??'?'} · ${session.lives_remaining??'?'} lives · ${title(phase.phase)} · risk ${title(risk.stance)}. ${top||'No model judgment yet'}. Raw model goal ${title(decision.model_goal)} → ${resolvedGoal}. Controller: ${title(payload.action)} — ${decision.reason||'pending'}. Model changed motor action: ${decision.model_changed_action?'yes':'no'}; objective changed motor action: ${decision.mode_changed_action?'yes':'no'}.`;
  if(planMode)$('situationSummary').textContent=`World ${session.world??'?'}-${session.stage??'?'} · ${session.lives_remaining??'?'} lives. Model plan: ${decision.model_plan||'DEFER'}. Context gate: ${decision.plan_intent?.reason||'pending'}. Model ranking ${decision.plan_model_applied?'APPLIED':'NOT APPLIED'}. Final authority: ${decision.arbitration?.selected||decision.source}. Action: ${title(payload.action)} — ${decision.reason}.`;
  const report={
    snapshot:{epoch:payload.epoch,frame:payload.frame_index,model_source:decision.source_snapshot||null,
              model_age_frames:timing.model_age_frames,model_in_flight:timing.in_flight},
    model:{kind:planMode?'plan':'goal',selected:(planMode?decision.model_plan:decision.model_goal)||null,
           goal_scores:planMode?null:Object.fromEntries(weights),plan_scores:planMode?Object.fromEntries(weights):null,model_id:decision.model_id||null,
           stack:isLaya?'LAYA ONNX judgment model; JevT++ is the C++ graph/controller':'Open-JEV adapter and decision head on pinned base model; JevT++ is the C++ graph/controller',
           base_model_id:baseModelId||null,decision_method:runtime.open_jev_method||null,
           cuda_device:runtime.validated?runtime.device:null,latency_ms:timing.model_wall_ms},
    goal_resolution:{raw_model_goal:decision.raw_model_goal||decision.model_goal||null,
                     selected_feasible_goal:decision.active_goal||null,
                     selection:decision.intent_selection||null,
                     selected_score:decision.selected_goal_probability??null},
    plan_judgment:planMode?{selected:decision.model_plan,scores:decision.plan_scores,defer:decision.defer_score,context:decision.plan_context,gate:decision.plan_intent,applied:decision.plan_model_applied}:null,
    arbitration:decision.arbitration||null,
    game_session:session,game_phase:phase,learning_delta:delta,risk_policy:risk,
    digital_twin:payload.digital_twin||null,
    online_learning:Object.fromEntries(['world_belief','action_dynamics','persistent_tasks','outcome_update','risk_budget','hunter_engagement'].map(id=>[id,payload.graph?.nodes?.find(n=>n.id===id)?.output??null])),
    controller:{action:payload.action,baseline_action:decision.baseline_action,
                objective_mode:payload.mode,reason:decision.reason,phase:decision.phase,source:decision.source,
                intent_accepted:decision.intent_accepted,strategy_applied:decision.strategy_applied,
                intent_gate:decision.intent_reason,mode_effect:decision.mode_effect,
                mode_changed_action:decision.mode_changed_action,
                model_effect:decision.model_effect,evidence:decision.evidence},
    observed:{ground_source:state.player?.ground_source,grounded:state.player?.grounded,
              movement_state:state.player?.movement_state,enemies:state.hazard?.upcoming_enemies||[]},
    learned_context:payload.knowledge?.context||null,
    limitations:'Online action-conditioned physics and task memory, not neural weight training. Model scores are not win probabilities; planner risk costs are not death probabilities.'
  };
  $('situationReport').textContent=JSON.stringify(report,null,2);
}
function renderLearner(payload){
  const output=id=>payload.graph?.nodes?.find(n=>n.id===id)?.output||{};
  const world=output('world_belief'),dyn=output('action_dynamics'),task=output('persistent_tasks'),risk=output('risk_budget'),events=output('outcome_update');
  const combat=output('hunter_engagement');
  const authority=payload.decision?.arbitration?.selected;
  $('learnerStatus').textContent=authority?`${title(authority)} CONTROLS`:payload.decision?.source==='jevtpp_hunter_controller'?'HUNTER CONTROLS':payload.decision?.planner_applied?'PLANNER CONTROLS':'BASELINE / GUARD';
  const values=[['OBSERVED / CONFIRMED',`${world.entities?.length??0} / ${world.confirmed_eliminations??0}`],['UNRESOLVED HOSTILES',world.unresolved??0],['TRANSITION SAMPLES',dyn.samples??0],['PREDICTION ERROR',dyn.prediction_error_px?.mean==null?'UNKNOWN':`${number(dyn.prediction_error_px.mean,2)} PX`],['RISK BUDGET (COST)',risk.budget??'—'],['FEASIBLE PLANS',risk.eligible_candidates??0]];
  $('learnerFacts').replaceChildren(...values.map(([label,value])=>{const el=document.createElement('div');el.className='fact';const small=document.createElement('small'),strong=document.createElement('strong');small.textContent=label;strong.textContent=value;el.append(small,strong);return el}));
  $('taskSummary').textContent=`${title(task.objective)} · ${title(task.clearance)}: ${task.confirmed_eliminations??0} confirmed, ${task.unresolved_targets??0} unresolved · ${title(task.status)} · target ${task.target?.id??'none'} · ${title(combat.apply?combat.phase:risk.reason)}. Latest feedback: ${(events.events||[]).map(e=>title(e.type)).join(', ')||'no new event'}.`;
  const flight=dyn.active_flight?.signature?dyn.active_flight:dyn.last_flight;
  if(flight?.signature)$('taskSummary').textContent+=` Flight: ${title(flight.result)} · takeoff ${flight.signature} · A ${dyn.jump_cut?'CUT (cannot restore height)':'NOT CUT'}.`;
  if(dyn.learning_supervisor)$('taskSummary').textContent+=` Learning: prediction-validated shadow estimates; parameter promotion is not a policy-win guarantee.`;
  const plans=(output('candidate_plans').candidates||[]).slice().sort((a,b)=>b.utility-a.utility).slice(0,5);
  $('planRows').replaceChildren(...plans.map(plan=>{const row=document.createElement('tr');if(plan.id===risk.selected?.id)row.className='selected';for(const value of [`${plan.id===risk.selected?.id?'▶ ':''}${plan.sequence.map(x=>`${x.action} ×${x.frames}`).join(' → ')}`,number(plan.utility,1),number(plan.risk_cost,2),plan.predicted_collision?'COLLISION':plan.continuation_failed?'NEXT TRANSFER FAILED':plan.unsafe_jump_cut?'UNSAFE INPUT CUT':plan.predicted_stomp?'PREDICTED STOMP':plan.unknown_space?'UNKNOWN SPACE':plan.supported_end?'SUPPORTED LANDING':'AIRBORNE']){const td=document.createElement('td');td.textContent=value;row.append(td)}return row}));
}
function drawKnowledge(payload){
  const memory=payload.knowledge||{},canvas=$('knowledgeMap'),ctx=canvas.getContext('2d'),d=devicePixelRatio||1;
  const twin=payload.digital_twin||payload.graph?.nodes?.find(node=>node.id==='digital_twin')?.output;
  const foe=twin?.enemy||{},landing=twin?.landing||{};
  const foeName=payload.state?.hazard?.nearest_enemy_kind||`TYPE ${foe.kind_id??'?'}`;
  $('twinSummary').textContent=twin?`${foe.present?`${title(foeName)} at ${number(foe.distance_px,0)} px · contact ${foe.contact_eta_frames==null?'unknown':number(foe.contact_eta_frames,1)+' fr'}`:'No current enemy contact predicted'} · landing ${landing.eta_frames==null?title(landing.status):number(landing.eta_frames,1)+' fr'} · horizon ${twin.horizon_frames??'—'} fr`:'Waiting for a current RAM projection.';
  $('twinProvenance').textContent=twin?`SOURCE ${title(twin.source)} · RELIABILITY ${title(twin.reliability)} · ENEMY VELOCITY ${title(foe.velocity_source||'unknown')} (${foe.velocity_samples||0} samples) · UNKNOWN GEOMETRY STAYS UNKNOWN`:'No forecast is available yet.';
  const learned=payload.learning_delta||{},policy=payload.risk_policy||{};
  $('learningSummary').textContent=`THIS FRAME: ${learned.new_columns??0} NEW COLUMNS · ${learned.new_enemy_types??0} ENEMY TYPES · ${learned.enemy_velocity_samples??0} VELOCITY SAMPLES · ${learned.jump_samples??0} JUMP SAMPLES · RISK ${title(policy.stance)}${policy.reasons?.length?' ('+policy.reasons.join(', ')+')':''}`;
  const w=canvas.clientWidth,h=220;
  if(canvas.width!==Math.round(w*d)||canvas.height!==h*d){canvas.width=Math.round(w*d);canvas.height=h*d}
  ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);
  const player=payload.state?.player||{},level=payload.state?.level||{world:1,stage:1};
  const columns=(memory.columns||[]).filter(c=>c.world===level.world&&c.stage===level.stage);
  const start=Math.min(0,...columns.map(c=>c.x)),end=Math.max(Number(player.x||0)+256,...columns.map(c=>c.x+16),256);
  const sx=(w-32)/Math.max(256,end-start),sy=(h-36)/240;
  ctx.fillStyle='#476553';ctx.font='10px monospace';ctx.fillText('KNOWN / REMEMBERED TILES · DARK SPACE = UNKNOWN',16,14);
  for(const column of columns){
    const current=!column.imported&&column.last_epoch===payload.epoch;
    ctx.fillStyle=current?'#358957':'#244131';
    for(const y of column.solid_y||[])ctx.fillRect(16+(column.x-start)*sx,24+y*sy,Math.max(1,16*sx-1),Math.max(2,16*sy-1));
    if(!column.solid_y?.length){ctx.fillStyle='#498799';ctx.fillRect(16+(column.x-start)*sx,h-10,Math.max(1,16*sx-1),2)}
  }
  const px=16+(Number(player.x||0)-start)*sx,py=24+Number(player.screen_y||0)*sy;
  ctx.strokeStyle='#65e097';ctx.strokeRect(Math.max(16,px-48*sx),22,256*sx,h-34);
  ctx.fillStyle='#ffd16c';ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fill();
  for(const enemy of payload.state?.hazard?.upcoming_enemies||[]){
    ctx.fillStyle='#fa7878';ctx.fillRect(16+(Number(enemy.x??(Number(player.x||0)+enemy.relative_x_pixels))-start)*sx,24+Number(enemy.screen_y||0)*sy,4,5);
  }
  const target=payload.decision?.landing_plan;
  const selected=payload.graph?.nodes?.find(n=>n.id==='risk_budget')?.output?.selected;
  if(selected?.path?.length){
    ctx.strokeStyle='#60d3ff';ctx.lineWidth=2;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(px,24+Number(player.feet_y||0)*sy);
    for(const point of selected.path)ctx.lineTo(16+(point.x-start)*sx,24+point.feet_y*sy);
    ctx.stroke();ctx.setLineDash([]);
  }
  if(target){
    const left=16+(Number(target.left_x)-start)*sx, right=16+(Number(target.right_x)-start)*sx, y=24+Number(target.y||0)*sy;
    ctx.strokeStyle='#f0ad27';ctx.lineWidth=2;ctx.strokeRect(left,y-4,Math.max(4,right-left),8);
  }
  const jumps=memory.jumps?.duration_frames||{},range=memory.jumps?.range_pixels||{},outcomes=memory.outcomes||{};
  $('knowledgeCount').textContent=`${columns.length} COLUMNS · ${jumps.count||0} JUMPS`;
  $('knowledgeSummary').textContent=`Observed samples: ${memory.observed_frames||0}. Completed jumps: ${jumps.count||0}; mean air time ${jumps.mean==null?'unknown':number(jumps.mean,1)+' frames'}, range ${range.mean==null?'unknown':number(range.mean,1)+' px'}. Enemy types: ${memory.enemy_types?.length||0}. Recorded clears/deaths: ${outcomes.successes||0}/${outcomes.failures||0}. ${payload.knowledge_scope==='session'?'Observed memory survives GAME OVER in this process.':'Memory resets at GAME OVER;'} neural weights are unchanged.`;
  $('knowledgeGraph').textContent=JSON.stringify({scope:memory.scope,context:memory.context,graph:memory.graph,enemy_types:memory.enemy_types},null,2);
}
function node(ctx,x,y,w,h,label,value,kind,active,key,accent){
  ctx.save(); const color=accent||(kind==='judgment'?'#f0ad27':'#3dff74');ctx.lineWidth=active?2:1; ctx.strokeStyle=active?color:'#183524'; ctx.fillStyle=active?'#0b1d11':'#07100b';
  ctx.shadowColor=active?ctx.strokeStyle:'transparent'; ctx.shadowBlur=active?10:0; ctx.beginPath(); ctx.roundRect(x,y,w,h,kind==='result'?0:9);ctx.fill();ctx.stroke();ctx.shadowBlur=0;
  ctx.fillStyle=active?ctx.strokeStyle:'#57715e';ctx.font='9px Cascadia Code, monospace';ctx.fillText(label.toUpperCase(),x+12,y+17);
  ctx.fillStyle=active?'#c9f8d3':'#46614e';ctx.font='600 11px Cascadia Code, monospace';ctx.fillText(String(value).slice(0,22),x+12,y+37);
  if(key&&(hoveredGraphNode===key||pinnedGraphNode===key)){ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.strokeRect(x-3,y-3,w+6,h+6)}
  ctx.restore();if(key)graphHitAreas.push({x,y,w,h,key});
}
function edge(ctx,a,b,active,color='#3dff74'){
  ctx.save();ctx.strokeStyle=active?color:'#102719';ctx.lineWidth=active?2:1;ctx.setLineDash(active?[]:[4,5]);ctx.beginPath();ctx.moveTo(a[0],a[1]);const mid=(a[0]+b[0])/2;ctx.bezierCurveTo(mid,a[1],mid,b[1],b[0],b[1]);ctx.stroke();ctx.restore();
}
function packet(ctx,a,b,active,color,offset=0){
  if(!active)return;const t=(performance.now()/1250+offset)%1,mid=(a[0]+b[0])/2,u=1-t;
  const x=u*u*u*a[0]+3*u*u*t*mid+3*u*t*t*mid+t*t*t*b[0];
  const y=u*u*u*a[1]+3*u*u*t*a[1]+3*u*t*t*b[1]+t*t*t*b[1];
  ctx.save();ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=12;ctx.beginPath();ctx.arc(x,y,3.1,0,Math.PI*2);ctx.fill();ctx.restore();
}
function flowLabel(ctx,x,y,text){ctx.save();ctx.fillStyle='#63836c';ctx.font='8px Cascadia Code, monospace';ctx.fillText(text.toUpperCase(),x,y);ctx.restore()}
let graphView={zoom:1,panX:0,panY:0}, graphTransform={scale:1,x:0,y:0};
let traceReceivedAt=0,lastTraceId='',dragStart=null;
function drawGraph(payload){
  const canvas=$('decisionGraph'),d=devicePixelRatio||1,w=canvas.clientWidth,h=520;
  if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){canvas.width=Math.round(w*d);canvas.height=Math.round(h*d)}
  const c=canvas.getContext('2d');c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,w,h);graphHitAreas=[];
  const trace=payload?.graph;
  if(!trace?.nodes?.length){c.fillStyle='#91aa98';c.font='12px monospace';c.fillText('Waiting for executed JevT++ discovery DAG trace',20,35);return}
  const depth={},columns=[];
  for(const item of trace.nodes){
    depth[item.id]=item.dependencies.length?1+Math.max(...item.dependencies.map(id=>depth[id]??0)):0;
    (columns[depth[item.id]]??=[]).push(item);
  }
  const nw=170,nh=68,gapX=55,gapY=35,worldW=columns.length*(nw+gapX)-gapX+40;
  const worldH=Math.max(...columns.map(col=>col.length))*(nh+gapY)+40;
  const scale=Math.min(w/worldW,h/worldH)*graphView.zoom;
  const ox=(w-worldW*scale)/2+graphView.panX,oy=(h-worldH*scale)/2+graphView.panY;
  graphTransform={scale,x:ox,y:oy};c.translate(ox,oy);c.scale(scale,scale);
  const positions={};
  columns.forEach((col,i)=>col.forEach((item,j)=>{positions[item.id]={x:20+i*(nw+gapX),y:30+j*(nh+gapY)+(worldH-col.length*(nh+gapY)-40)/2}}));
  const moving=payload.status==='running'&&performance.now()-traceReceivedAt<650;
  for(const [index,link] of trace.edges.entries()){
    const a=positions[link.from],b=positions[link.to];if(!a||!b)continue;
    const from=[a.x+nw,a.y+nh/2],to=[b.x,b.y+nh/2],color=link.from==='model_mailbox'?'#f0ad27':'#3dff74';
    edge(c,from,to,link.active,color);packet(c,from,to,link.active&&moving,color,index*.13);
    flowLabel(c,(from[0]+to[0])/2-14,(from[1]+to[1])/2-7,link.bytes+' B');
  }
  for(const item of trace.nodes){
    const p=positions[item.id],value=item.output||{};
    const summaries={plan_candidates:`${value.plan_candidates?.length??0} MODEL OPTIONS`,plan_intent_gate:value.accepted?'CONTEXT VERIFIED':title(value.reason),world_belief:`${value.confirmed_eliminations??0} KILLS / ${value.unresolved??0} OPEN`,action_dynamics:`${value.samples??0} TRANSITIONS`,persistent_tasks:title(value.status),candidate_plans:`${value.candidates?.length??0} ROLLOUTS`,risk_budget:`${value.eligible_candidates??0} FEASIBLE · ${value.budget??'?'}`,outcome_update:`${value.events?.length??0} EVENTS`,reactive_baseline:title(value.action),hunter_engagement:title(value.phase)};
    const text=summaries[item.id]??(item.id==='controller'?title(value.action):item.id==='intent_gate'?(value.accepted?'ACCEPTED':'REFLEX FALLBACK'):
      item.id==='model_mailbox'?(value.cached_response?.model_plan||value.cached_response?.active_goal||'MODEL PENDING'):
      item.id==='compact_state'?value.mode:
      item.id==='game_phase'?value.phase:
      item.id==='risk_policy'?value.stance:
      item.id==='learning_delta'?`${value.new_columns??0} NEW COL`:item.status);
    const color=item.status!=='succeeded'?'#ff7777':item.id==='model_mailbox'?'#f0ad27':
      (item.id==='intent_gate'||item.id==='plan_intent_gate')&&!value.accepted?'#f0ad27':'#3dff74';
    node(c,p.x,p.y,nw,nh,item.id,text,'state',item.status==='succeeded',item.id,color);
    c.fillStyle='#839d8c';c.font='9px monospace';c.fillText(number(item.duration_ms,3)+' ms · '+item.status,p.x+12,p.y+56);
  }
  c.setTransform(d,0,0,d,0,0);
  c.fillStyle='#839d8c';c.font='10px monospace';c.fillText('SNAPSHOT '+trace.snapshot_id+' · C++ DAG '+number(trace.duration_ms,3)+' ms'+(payload.graph_current===false?' · LAST PRE-TERMINAL DECISION':''),14,h-12);
}
function inspectorValue(key,payload){const trace=payload?.graph,record=trace?.nodes?.find(node=>node.id===key);if(record)return {...record,dependency_inputs:Object.fromEntries(record.dependencies.map(id=>[id,trace.nodes.find(n=>n.id===id)?.output]))};const s=payload?.state||{},d=payload?.decision||{};if(String(key).startsWith('check:'))return (s.checker_bus?.checks||[]).find(check=>check.id===key.slice(6))||{};return ({player:s.player,trajectory:s.trajectory,terrain:s.terrain,hazard:s.hazard,recent:s.recent_control,timing:s.reaction_timing,checkers:s.checker_bus,telemetry:{frame:payload.frame_index,decision_age_frames:payload.decision_age_frames,frames_until_decision:payload.frames_until_decision,strategy:s.strategy,detectors:s.detectors,optimization:s.optimization},goal:{selected:d.active_goal,confidence:d.goal_confidence,probabilities:d.goal_probabilities},choice:{raw_action:d.raw_action,confidence:d.raw_confidence,probabilities:d.probabilities},noul:{jump_probability:d.jump_probability,eligible_for_policy_gate:d.noul_gate_eligible,signal_used:d.jump_signal_used},score:{danger_score:d.danger_score,signal_used:d.jump_signal_used},gate:{override_applied:d.override_applied,reason:d.gate_reason,raw_action:d.raw_action,final_action:d.action},controller:{action:payload.action,frame:payload.frame_index}})[key]||s}
function updateInspector(){if(!lastPayload)return;const key=hoveredGraphNode||pinnedGraphNode||'telemetry';$('inspectTitle').textContent=title(key);$('inspectPayload').textContent=JSON.stringify(inspectorValue(key,lastPayload),null,2)}

let modelOptionsSignature='';
function renderModelSelection(payload){
  const selection=payload.model_selection;if(!selection)return;
  const signature=JSON.stringify(selection.options);
  if(signature!==modelOptionsSignature){
    $('modelSelect').replaceChildren(...selection.options.map(item=>{const option=document.createElement('option');option.value=item.id;option.disabled=!item.available;option.textContent=item.label+(item.available?'':' · unavailable');option.title=item.reason;return option}));
    modelOptionsSignature=signature;
  }
  if(document.activeElement!==$('modelSelect'))$('modelSelect').value=selection.selected;
  const unavailable=selection.options.filter(item=>!item.available);
  $('modelSelectStatus').textContent=unavailable.length?unavailable.map(item=>`${item.label}: ${item.reason}`).join(' · '):'CUDA · same C++ controller';
}
function update(payload){if(payload.graph?.snapshot_id!==lastTraceId){traceReceivedAt=performance.now();lastTraceId=payload.graph?.snapshot_id}lastPayload=payload;renderModelSelection(payload);const decision=payload.decision||{};$('runStatus').textContent=title(payload.status);$('decisionIndex').textContent=String(payload.decision_index||0).padStart(4,'0');$('frameIndex').textContent=payload.frame_index||0;$('backend').textContent=decision.backend||'JEVT++';$('latency').textContent=decision.latency_ms==null?'—':`${number(decision.latency_ms,1)} MS`;$('activeAction').textContent=title(payload.action);$('reward').textContent=`${Number(payload.episode_reward||0)>=0?'+':''}${number(payload.episode_reward,1)}`;const topGoalWeight=Math.max(0,...Object.values(decision.goal_probabilities||{}).map(Number).filter(Number.isFinite));$('confidence').textContent=`TOP GOAL ${number(topGoalWeight*100,1)}%`;
  $('activeGoal').textContent=`${title(decision.model_goal||'waiting for model')} → ${decision.intent_accepted?title(decision.active_goal):'REFLEX'}`;if(document.activeElement!==$('modeSelect'))$('modeSelect').value=payload.mode||'speedrun';
  if(document.activeElement!==$('knowledgeScope'))$('knowledgeScope').value=payload.knowledge_scope||'run';
  $('dashboardVersion').textContent=`${String(payload.version||'unknown').split('-')[0]} · ${String(payload.version).includes('knowledge-retention')?'KNOWLEDGE LOOP':payload.version==='v14-generic-skills'?'GENERIC SKILLS':'LEGACY PIPELINE'}`;
  if(decision.judgment_kind==='plan'){
    $('activeGoal').textContent=`${title(decision.candidate_labels?.[decision.model_plan]||decision.model_plan||'DEFER')} → ${decision.plan_intent_accepted?'VERIFIED':'LOCAL CONTROL'}`;
    const score=Math.max(0,decision.defer_score||0,...Object.values(decision.plan_scores||{}));$('confidence').textContent=`TOP OPTION ${number(score*100,1)}%`;
  }
  $('gateReason').textContent=`${decision.source||'JevT++'} · ${decision.reason||'pending'} · OBJECTIVE ${title(payload.mode)}: ${title(decision.mode_effect||'no motor effect')} · MODEL ${decision.strategy_applied?'APPLIED':'NOT APPLIED'} · ${decision.intent_reason||''}`;
  if(decision.judgment_kind==='plan')$('gateReason').textContent=`ARBITER ${decision.arbitration?.selected||'pending'} · ${decision.reason||'pending'} · MODEL RANK ${decision.plan_model_applied?'APPLIED':'NOT APPLIED'} · ${decision.plan_intent?.reason||''}`;
  $('flowFrame').textContent=payload.frame_index??'—';$('decisionAge').textContent=payload.decision_age_frames==null?'PENDING':`${payload.decision_age_frames} FR`;
  $('nextBatch').textContent=payload.timing?.in_flight?'IN FLIGHT':`${payload.frames_until_decision??0} FR`;
  const jump=String(payload.action).includes('jump')?1:0,ev=decision.evidence||{};
  const nearest=Math.min(...['gap_distance_pixels','wall_distance_pixels','enemy_distance_pixels'].map(k=>ev[k]??999));
  const danger=clamp(1-nearest/100);
  $('jumpMeter').style.width=`${jump*100}%`;$('jumpValue').textContent=jump?'A HELD':'RELEASED';
  $('dangerMeter').style.width=`${danger*100}%`;$('dangerValue').textContent=nearest<999?`${nearest} PX`:'CLEAR';
  $('controlLatency').textContent=`${number(payload.timing?.control_ms,2)} MS`;
  $('emulatorFps').textContent=`${number(payload.timing?.emulated_fps,1)} FPS`;
  $('modelError').textContent=payload.timing?.model_error||'';
  paused=payload.status==='paused';$('pauseButton').textContent=paused?'RESUME':'PAUSE';
  renderProbabilities(decision);renderState(payload);renderSituation(payload);renderLearner(payload);drawKnowledge(payload);drawGraph(payload);if(typeof renderOntology==='function')renderOntology(payload);updateInspector();
}
async function poll(){try{const r=await fetch('/api/live',{cache:'no-store'});if(r.ok)update(await r.json())}catch(e){$('runStatus').textContent='OFFLINE'}}
setInterval(()=>{$('gameFrame').src=`/api/frame?t=${Date.now()}`},90);setInterval(poll,180);poll();
async function command(value){await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:value})});}
$('pauseButton').addEventListener('click',()=>command(paused?'resume':'pause'));
$('restartButton').addEventListener('click',()=>command('restart'));
$('modeSelect').addEventListener('change',event=>fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'mode',mode:event.target.value})}));
$('knowledgeScope').addEventListener('change',async event=>{
  const select=event.target;
  try{const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'knowledge_scope',scope:select.value})});
    if(!response.ok)throw new Error('Knowledge mode change rejected');}
  catch(error){select.value=lastPayload?.knowledge_scope||'run';$('modelError').textContent=error.message;}
});
$('modelSelect').addEventListener('change',async event=>{
  const select=event.target;select.disabled=true;
  try{const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'model',model:select.value})});if(!response.ok)throw new Error('Model unavailable; current backend retained');$('modelSelectStatus').textContent='Switching judgment backend…';}
  catch(error){select.value=lastPayload?.model_selection?.selected||'open_jev';$('modelSelectStatus').textContent=error.message;}
  finally{select.disabled=false;}
});
addEventListener('resize',()=>lastPayload&&drawGraph(lastPayload));
const graph=$('decisionGraph');
function graphHit(event){const rect=graph.getBoundingClientRect(),x=(event.clientX-rect.left-graphTransform.x)/graphTransform.scale,y=(event.clientY-rect.top-graphTransform.y)/graphTransform.scale;return graphHitAreas.find(hit=>x>=hit.x&&x<=hit.x+hit.w&&y>=hit.y&&y<=hit.y+hit.h)?.key||null}
graph.addEventListener('pointerdown',event=>{hoveredGraphNode=graphHit(event);dragStart={x:event.clientX,y:event.clientY,px:graphView.panX,py:graphView.panY};graph.setPointerCapture(event.pointerId)});
graph.addEventListener('pointermove',event=>{
  if(dragStart){graphView.panX=dragStart.px+event.clientX-dragStart.x;graphView.panY=dragStart.py+event.clientY-dragStart.y}
  const rect=graph.getBoundingClientRect(),x=(event.clientX-rect.left-graphTransform.x)/graphTransform.scale,y=(event.clientY-rect.top-graphTransform.y)/graphTransform.scale;
  hoveredGraphNode=graphHitAreas.find(hit=>x>=hit.x&&x<=hit.x+hit.w&&y>=hit.y&&y<=hit.y+hit.h)?.key||null;
  graph.style.cursor=dragStart?'grabbing':hoveredGraphNode?'pointer':'grab';updateInspector();
});
graph.addEventListener('pointerup',event=>{
  if(dragStart&&Math.hypot(event.clientX-dragStart.x,event.clientY-dragStart.y)<5&&hoveredGraphNode){pinnedGraphNode=hoveredGraphNode;updateInspector()}
  dragStart=null;
});
graph.addEventListener('pointercancel',()=>{dragStart=null});
graph.addEventListener('pointerleave',()=>{if(!dragStart)hoveredGraphNode=null});
$('graphZoomIn').addEventListener('click',()=>{graphView.zoom=Math.min(3,graphView.zoom*1.2)});
$('graphZoomOut').addEventListener('click',()=>{graphView.zoom=Math.max(.5,graphView.zoom/1.2)});
$('graphFit').addEventListener('click',()=>{graphView={zoom:1,panX:0,panY:0}});
graph.addEventListener('keydown',event=>{
  const nodes=lastPayload?.graph?.nodes||[],index=nodes.findIndex(n=>n.id===pinnedGraphNode);
  if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();pinnedGraphNode=nodes[(index+(event.key==='ArrowRight'?1:-1)+nodes.length)%nodes.length]?.id;updateInspector()}
});
function animateGraph(){if(lastPayload)drawGraph(lastPayload);requestAnimationFrame(animateGraph)}
requestAnimationFrame(animateGraph);
