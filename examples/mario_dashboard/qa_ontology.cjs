const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const path=require('node:path');
(async()=>{
  const browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1600,height:1000}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  try{
    const root=process.env.JEVT_DEMO_URL||'http://127.0.0.1:4173';
    await page.goto(root+'/?dashboard=v16',{waitUntil:'domcontentloaded'});
    const initial=await (await page.request.get(root+'/api/live')).json();
    await page.waitForFunction(async()=>{
      const current=await (await fetch('/api/live')).json();
      return current.ontology?.nodes?.length>4&&current.ontology?.edges?.length>3;
    },null,{timeout:15000});
    const live=await (await page.request.get(root+'/api/live')).json();
    assert.equal(live.ontology.schema,'jevt.knowledge_view.v1');
    assert(live.ontology.nodes.length>4);
    assert(live.ontology.edges.length>3);
    assert(live.ontology.edges.every(e=>live.ontology.nodes.some(n=>n.id===e.from)&&
      live.ontology.nodes.some(n=>n.id===e.to)));
    await page.locator('#ontologyCanvas').focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(()=>document.querySelector('#ontologyInspect').textContent.includes('"relations"'));
    await page.locator('#ontologyFilter').selectOption('empirical');
    assert.equal(await page.locator('#ontologyFilter').inputValue(),'empirical');
    if(live.ontology.edges.some(e=>e.relation==='observed_failure_context')){
      await page.locator('#ontologyFilter').selectOption('feedback');
      assert.equal(await page.locator('#ontologyFilter').inputValue(),'feedback');
      assert.match(await page.locator('#ontologyCount').textContent(),/SHOWN/);
    }
    await page.locator('#ontologyFilter').selectOption('all');
    await page.locator('.ontology-panel').screenshot({path:path.join(__dirname,'artifacts','ontology-v16-feedback-1600.png')});
    await page.setViewportSize({width:390,height:844});
    const mobile=await page.evaluate(()=>({width:document.documentElement.scrollWidth,
      panel:document.querySelector('.ontology-panel').getBoundingClientRect().width,
      graph:document.querySelector('.ontology-scroll').scrollWidth,
      visible:document.querySelector('.ontology-scroll').clientWidth}));
    assert(mobile.width<=392);
    assert(mobile.graph>mobile.visible);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({nodes:live.ontology.nodes.length,edges:live.ontology.edges.length,
      focus:live.ontology.model_focus,mobile,errors},null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
