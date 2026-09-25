// Isolated browser QA: model switching must not restart the NES or its memory.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const root=process.env.JEVT_DEMO_URL||'http://127.0.0.1:4173';
 const live=async()=>await(await page.request.get(root+'/api/live')).json();
 const command=async value=>page.request.post(root+'/api/command',{data:{command:value}});
 try{
  await page.goto(root+'/?dashboard=v14');
  await page.waitForFunction(()=>document.querySelector('#modelSelectStatus').textContent.includes('same C++'));
  const original=await live();
  assert.equal(original.version,'v14-generic-skills');
  assert.equal(original.status,'running','Start QA while game is active; this test never resets a run');
  await command('pause');
  await page.waitForFunction(()=>document.querySelector('#runStatus').textContent==='PAUSED');
  const before=await live();
  await page.selectOption('#modelSelect','laya');
  await page.waitForFunction(()=>document.querySelector('#modelStack').textContent.startsWith('LAYA'));
  const switched=await live();
  assert.equal(switched.model_selection.selected,'laya');
  assert.equal(switched.frame_index,before.frame_index);
  for(const key of ['run_id','world','stage','lives_remaining','memory_generation','session_frame'])
   assert.equal(switched.session[key],before.session[key],key+' changed on model switch');
  assert.equal(switched.graph_current,false);
  assert.equal(switched.decision.model_id,undefined,'Old model result leaked across switch');
  await command('resume');
  await page.waitForFunction(()=>document.querySelector('#backend').textContent.includes('Laya CUDA'),{},{timeout:20000});
  const inferred=await live();
  assert.match(inferred.decision.model_id,/laya/i);
  assert.equal(inferred.decision.judgment_kind,'plan');
  assert.equal(inferred.decision.slot_scores.length,7);
  assert.equal(inferred.decision.epoch,inferred.epoch);
  assert.equal(inferred.runtime.provider,'CUDAExecutionProvider');
  assert.equal(inferred.runtime.device,'cuda:0');
  await page.screenshot({path:path.join(__dirname,'artifacts','model-laya-v14.png'),fullPage:false});
  await command('pause');
  await page.waitForFunction(()=>document.querySelector('#runStatus').textContent==='PAUSED');
  await page.selectOption('#modelSelect','open_jev');
  await page.waitForFunction(()=>document.querySelector('#modelStack').textContent.startsWith('OPEN-JEV'));
  await command('resume');
  await page.waitForFunction(()=>document.querySelector('#backend').textContent.includes('Open-JEV model'),{},{timeout:20000});
  const restored=await live();
  assert.equal(restored.runtime.model_kind,'open_jev');
  assert.equal(restored.decision.judgment_kind,'plan');
  assert.equal(restored.decision.epoch,restored.epoch);
  assert.equal(restored.session.run_id,before.session.run_id);
  assert.deepEqual(errors,[]);
  const report={paused_switch_preserved_session:true,laya:{id:inferred.decision.model_id,latency_ms:inferred.decision.latency_ms,runtime:inferred.runtime},
   open_jev:{id:restored.decision.model_id,latency_ms:restored.decision.latency_ms},errors};
  fs.writeFileSync(path.join(__dirname,'artifacts','model-switch-v14.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
