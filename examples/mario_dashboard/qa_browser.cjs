// Fresh isolated Chromium, only local demo; no user browser profile.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const artifacts = path.join(__dirname, 'artifacts');
  fs.mkdirSync(artifacts, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const root = process.env.JEVT_DEMO_URL || 'http://127.0.0.1:4173';
  const report = { url: root + '/?dashboard=v14', viewports: [], errors: [] };
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto(report.url, { waitUntil: 'domcontentloaded', timeout: 15000 });
    await page.waitForFunction(() => document.querySelector('#runStatus').textContent !== 'CONNECTING');
    await page.waitForFunction(() => document.querySelector('#gpuDevice').textContent.includes('4090'));
    await page.waitForFunction(() => document.querySelector('#gameFrame').naturalWidth > 0);
    const live = await (await page.request.get(root + '/api/live')).json();
    assert.equal(live.version, 'v14-generic-skills');
    assert.equal(await page.locator('#modelSelect').inputValue(), 'open_jev');
    assert.equal(live.runtime.validated, true);
    assert.equal(live.runtime.device, 'cuda:0');
    assert.equal(live.runtime.base_model_id, 'Qwen/Qwen3.5-2B');
    assert.equal(live.digital_twin.source, 'current_ram_projection');
    assert.ok(live.graph.nodes.some(node => node.id === 'digital_twin'));
    for (const id of ['game_phase', 'learning_delta', 'risk_policy','world_belief','action_dynamics','persistent_tasks','candidate_plans','plan_candidates','plan_intent_gate','risk_budget','outcome_update','hunter_engagement'])
      assert.ok(live.graph.nodes.some(node => node.id === id), `missing executed node ${id}`);
    assert.equal(live.env, 'SuperMarioBros-v0');
    assert.ok(Number.isInteger(live.session.lives_remaining));
    assert.ok((await page.locator('#stagesCleared').textContent()).includes('/ 32'));
    assert.ok((await page.locator('#modelStack').textContent()).includes('Qwen/Qwen3.5-2B'));
    assert.ok((await page.locator('#twinSummary').textContent()).includes('horizon'));
    report.live = { status: live.status, frame: live.frame_index, gpu: live.runtime.device_name,
                    nodes: live.graph.nodes.length, model_ms: live.timing.model_wall_ms,
                    twin_reliability: live.digital_twin.reliability,
                    world: `${live.session.world}-${live.session.stage}`,
                    lives: live.session.lives_remaining,
                    memory_generation: live.session.memory_generation,
                    fps: live.timing.emulated_fps, jump_samples: live.knowledge.context.jump_samples };
    await page.locator('#decisionGraph').focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('#inspectPayload').textContent.includes('dependencies'));
    await page.locator('#graphZoomIn').click();
    await page.locator('#graphFit').click();
    for (const size of [{ width: 1600, height: 1000 }, { width: 900, height: 900 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(size);
      await page.waitForTimeout(300);
      const geometry = await page.evaluate(() => ({
        width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
        headerClipping: [...document.querySelectorAll('.topbar select, .topbar button, .topbar label')].some(node => {
          const box = node.getBoundingClientRect(), header = document.querySelector('.topbar').getBoundingClientRect();
          return box.top < header.top || box.bottom > header.bottom;
        }),
        overflow: [...document.querySelectorAll('body *')].map(node => ({ tag: node.tagName,
          class: typeof node.className === 'string' ? node.className : '',
          id: node.id, right: Math.round(node.getBoundingClientRect().right) }))
          .filter(node => node.right > innerWidth + 2).slice(0, 15),
        game: { width: document.querySelector('#gameFrame').naturalWidth, height: document.querySelector('#gameFrame').naturalHeight },
        visible: [...document.querySelectorAll('.panel, .topbar')].map(node => {
          const box = node.getBoundingClientRect();
          return { class: node.className, left: box.left, right: box.right, width: box.width };
        })
      }));
      report.viewports.push({ ...size, ...geometry });
      await page.screenshot({ path: path.join(artifacts, 'dashboard-v14-' + size.width + '.png'), fullPage: true });
      if (size.width === 1600) {
        await page.locator('.graph-panel').screenshot({ path: path.join(artifacts, 'dashboard-v13-graph.png') });
        await page.locator('.knowledge-panel').screenshot({ path: path.join(artifacts, 'dashboard-v13-memory.png') });
        await page.locator('.online-panel').screenshot({ path: path.join(artifacts, 'dashboard-v13-learning.png') });
      }
    }
    fs.writeFileSync(path.join(artifacts, 'dashboard-browser-qa.json'), JSON.stringify(report, null, 2));
    assert.equal(report.errors.length, 0, JSON.stringify(report.errors));
    assert.ok(report.viewports.every(view => !view.headerClipping), 'Header controls clipped');
    const overflow = report.viewports.filter(view => view.scrollWidth > view.width + 2);
    assert.equal(overflow.length, 0, 'horizontal overflow: ' + JSON.stringify(overflow));
    console.log(JSON.stringify(report, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
