// DOM/canvas contract smoke test. This verifies JS behavior, not visual layout.
// Run with: node examples/mario_dashboard/test_dashboard.cjs
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, 'web');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const source = fs.readFileSync(path.join(root, 'dashboard.js'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
assert.equal(new Set(ids).size, ids.length, 'HTML contains duplicate IDs');

function createHarness() {
  let now = 1000;
  const requests = [], texts = [], rectangles = [], intervals = [], animationFrames = [];
  const drawing = {};
  for (const method of ['save', 'restore', 'beginPath', 'roundRect', 'fill', 'stroke', 'strokeRect',
                        'setLineDash', 'moveTo', 'bezierCurveTo', 'arc', 'setTransform', 'clearRect',
                        'translate', 'scale', 'lineTo']) drawing[method] = () => {};
  drawing.fillText = text => texts.push(String(text));
  drawing.fillRect = function(x, y, width, height) {
    rectangles.push({x, y, width, height, color: this.fillStyle});
  };
  const elements = new Map(ids.map(id => [id, {
    id, textContent: '', innerHTML: '', value: '', style: {}, dataset: {}, listeners: {},
    clientWidth: 1200, width: 0, height: 0,
    classList: {values: new Set(), toggle(name, force) { force ? this.values.add(name) : this.values.delete(name); }},
    addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); },
    dispatch(type, detail = {}) {
      const event = {target: this, clientX: 0, clientY: 0, pointerId: 1, preventDefault() {}, ...detail};
      return Promise.all((this.listeners[type] || []).map(handler => handler(event)));
    },
    getContext(kind) { assert.equal(kind, '2d'); return drawing; },
    getBoundingClientRect() { return {left: 0, top: 0, width: 1200, height: 520}; },
    setPointerCapture() {},
    replaceChildren(...children) { this.children = children; },
  }]));
  const sandbox = {
    console, performance: {now: () => now}, devicePixelRatio: 2,
    document: {activeElement: null, createElement(tag) { return {tag,style:{},children:[],textContent:'',append(...children){this.children.push(...children)}}; }, getElementById(id) {
      assert(elements.has(id), `dashboard requested missing HTML target: ${id}`);
      return elements.get(id);
    }},
    addEventListener() {},
    setInterval(handler, milliseconds) { intervals.push({handler, milliseconds}); return intervals.length; },
    requestAnimationFrame(handler) { animationFrames.push(handler); },
    fetch: async (url, options = {}) => {
      requests.push({url, options});
      return {ok: false}; // initial poll must not overwrite the explicit fixture
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, {filename: 'dashboard.js'});
  return {sandbox, elements, requests, texts, rectangles, intervals, animationFrames,
          advance(ms) { now += ms; },
          update(payload) { sandbox.update(payload); },
          inspect(expression) { return vm.runInContext(expression, sandbox); }};
}

function fixture() {
  const state = {
    level: {world: 1, stage: 1},
    strategy: {mode: 'speedrun'},
    player: {x: 580, y: 79, screen_y: 176, vx: 3, vy: 0, grounded: true,
             ground_source: 'ram_0x1d', movement_state: 0, jump_phase: 'grounded'},
    terrain: {gap_distance: 10, gap_width_tiles_visible: 3, gap_ahead: true,
              obstacle_distance: 999, observation_reliability: 'high'},
    hazard: {enemy_distance: 160, nearest_enemy_kind: 'goomba', estimated_contact_frames: 40,
             upcoming_enemies: [{slot: 0, kind_id: 6, kind: 'goomba', category: 'hostile',
               x: 740, screen_y: 184, relative_x_pixels: 160, relative_y_pixels: 8}]},
    recent_control: {outcome: 'forward', frames_observed: 1},
    detectors: {hostiles_visible: 1, powerups_visible: 0, powerup_distance: 999, coins: 2},
    optimization: {score: 1200, progress_pixels_per_second: 180},
    episode: {progress: 580, time_left: 365, lives: 2, dead: false},
  };
  const probabilities = {finish_fast: .15, stomp_enemy: .7, collect_powerup: .1,
                         collect_coins: .01, score_attack: .02, recover_momentum: .02};
  const cached = {ok: true, active_goal: 'stomp_enemy', epoch: 3, source_frame: 90,
                  source_snapshot: '3:90', goal_probabilities: probabilities};
  const context = {
    source: 'empirical_telemetry_memory', known_columns_ahead: 1,
    remembered_empty_columns_ahead: 1, jump_samples: 4, jump_duration_mean_frames: 36,
    jump_range_mean_pixels: 113, enemy_speeds: {'6': {mean: -1, samples: 11}},
    stage_successes: 9, deaths: 1, control_adaptation: false,
  };
  const stats = (count, mean, min, max) => ({count, mean, min, max, stddev: 1, m2: count - 1});
  const column = (x, solid_y, epoch, stage = 1) => ({world: 1, stage, x, solid_y,
    observations: 5, changes: 0, last_epoch: epoch, last_frame: 120, seen_order: x,
    confidence: .75, confidence_kind: 'observation-support heuristic, not a calibrated probability',
    source: 'visible_block_ram'});
  const knowledge = {
    schema: 'jevtpp_mario_experience', version: 1, environment: 'SuperMarioBros-1-1-v0',
    epoch: 3, frame: 120, observed_frames: 120,
    columns: [column(160, [208, 224], 2), column(576, [208, 224], 3),
              column(592, [], 3), column(1024, [32], 3, 2)],
    enemy_types: [{kind_id: 6, name: 'goomba', observations: 12,
                   velocity_x_px_per_frame: stats(11, -1, -1, -1)}],
    jumps: {duration_frames: stats(4, 36, 35, 37), range_pixels: stats(4, 113, 110, 116),
            height_pixels: stats(4, 62, 60, 64), hold_frames: stats(4, 30, 29, 31),
            completed_flights: 4, bounce_flights_excluded: 0},
    outcomes: {successes: 9, failures: 1}, context,
    scope: 'Observed RAM only. No unseen topology or automatic policy learning.',
    graph: {
      nodes: [{id: 'experience', source: 'telemetry'}, {id: 'observed_map', count: 4},
              {id: 'jump_estimate', count: 4}, {id: 'outcomes', successes: 9, failures: 1},
              {id: 'memory_context', control_adaptation: false}, {id: 'enemy_type_6', label: 'goomba', samples: 11}],
      edges: [{from: 'experience', to: 'observed_map', kind: 'observed'},
              {from: 'observed_map', to: 'memory_context', kind: 'summarized'},
              {from: 'experience', to: 'enemy_type_6', kind: 'observed'},
              {from: 'enemy_type_6', to: 'memory_context', kind: 'estimated_velocity'}],
    },
  };
  const twin = {
    source: 'current_ram_projection', horizon_frames: 8, reliability: 'current_ram',
    enemy: {present: true, kind_id: 6, distance_px: 160, velocity_px_per_frame: -1,
            velocity_source: 'current_ram', velocity_samples: 1, closing_speed_px_per_frame: 4,
            contact_eta_frames: 40, projected_distance_px: 128},
    landing: {status: 'grounded', eta_frames: 0, projected_x: 580, source: 'observed_ground', samples: 4},
    limits: ['Constant-velocity projection; not a collision simulation.'],
  };
  const outputs = {
    telemetry: state,
    model_mailbox: {cached_response: cached, epoch: 3, frame: 120},
    physics: {player: state.player, recent_control: state.recent_control, episode: state.episode},
    geometry: {terrain: state.terrain}, threats: {hazard: state.hazard},
    objectives: {strategy: state.strategy, detectors: state.detectors},
    experience_memory: context,
    digital_twin: twin,
    compact_state: {mode: 'speedrun', enemy_distance: 160, gap_distance: 10, experience: context},
    intent_gate: {accepted: false, active_goal: '', source_snapshot: '3:90', reason: 'reactive safety'},
    controller: {action: 'right_run_jump', active_goal: '', intent_accepted: false},
  };
  const dependencies = {
    telemetry: [], model_mailbox: [], physics: ['telemetry'], geometry: ['telemetry'],
    threats: ['telemetry'], objectives: ['telemetry'],
    experience_memory: ['telemetry'],
    digital_twin: ['physics', 'geometry', 'threats', 'experience_memory'],
    compact_state: ['physics', 'geometry', 'threats', 'objectives', 'experience_memory', 'digital_twin'],
    intent_gate: ['model_mailbox', 'compact_state'],
    controller: ['physics', 'geometry', 'threats', 'objectives', 'intent_gate'],
  };
  const nodes = Object.entries(dependencies).map(([id, dependencyIds]) => ({
    id, dependencies: dependencyIds, status: 'succeeded', output: outputs[id],
    snapshot_id: '3:120', duration_ms: .015, message: '',
  }));
  const edges = nodes.flatMap(node => node.dependencies.map(from => ({from, to: node.id, active: true, bytes: 140})));
  return {
    version: 'v9-async-dag', status: 'running', epoch: 3, frame_index: 120,
    decision_index: 4, action: 'right_run_jump', episode_reward: 50, mode: 'speedrun',
    decision_age_frames: 30, frames_until_decision: 0, state,
    runtime: {validated: true, cuda_available: true, device: 'cuda:0', device_name: 'NVIDIA GeForce RTX 4090'},
    knowledge, digital_twin: twin,
    debug: {local_grid: ['...M....', '#####...'], trusted_grid: []},
    decision: {backend: 'Open-JEV fixture', source: 'jevtpp_reactive_controller',
               model_goal: 'stomp_enemy', model_id: 'Open-JEV-2B-fixture', source_snapshot: '3:90',
               active_goal: '', goal_probabilities: probabilities, phase: 'takeoff', intent_accepted: false,
               goal_confidence: .41, latency_ms: 320, reason: 'gap', strategy_applied: false,
               intent_reason: 'reactive safety', evidence: {gap_distance_pixels: 10,
               wall_distance_pixels: null, enemy_distance_pixels: 160}},
    timing: {control_ms: .6, emulated_fps: 59.8, in_flight: true, model_error: null,
             model_age_frames: 30, model_wall_ms: 320},
    graph: {snapshot_id: '3:120', nodes, edges, duration_ms: .12},
  };
}

const tests = [];
function test(name, body) { tests.push({name, body}); }

test('all static JS targets exist in HTML', () => {
  for (const [, id] of source.matchAll(/\$\('([^']+)'\)/g)) assert(ids.includes(id), `missing #${id}`);
});

test('valid v9 payload renders actual model weights separately from control', () => {
  const h = createHarness(); h.update(fixture());
  assert.equal(h.elements.get('activeGoal').textContent, 'STOMP ENEMY → REFLEX');
  assert.equal(h.elements.get('activeAction').textContent, 'RIGHT RUN JUMP');
  assert.equal(h.elements.get('jumpValue').textContent, 'A HELD');
  assert.equal(h.elements.get('dangerValue').textContent, '10 PX');
  assert.equal(h.elements.get('latency').textContent, '320.0 MS');
  assert.equal(h.elements.get('controlLatency').textContent, '0.60 MS');
  const weights = h.elements.get('probabilities').innerHTML;
  assert.match(weights, /class="prob-row selected"><span>STOMP ENEMY<\/span>/);
  assert.match(weights, /70\.0%/);
  assert.doesNotMatch(weights, /RIGHT RUN JUMP/);
  assert.match(h.elements.get('gateReason').textContent, /MODEL NOT APPLIED/);
  assert.equal(h.elements.get('decisionGraph').width, 2400);
  assert.equal(h.elements.get('decisionGraph').height, 1040);
  assert(h.texts.some(text => text.includes('SNAPSHOT 3:120')));
});

test('keyboard inspector includes exact predecessor payloads and source snapshot', async () => {
  const h = createHarness(); h.update(fixture());
  await h.elements.get('decisionGraph').dispatch('keydown', {key: 'ArrowRight'});
  assert.equal(h.elements.get('inspectTitle').textContent, 'MODEL MAILBOX');
  let inspected = JSON.parse(h.elements.get('inspectPayload').textContent);
  assert.equal(inspected.output.cached_response.source_snapshot, '3:90');
  await h.elements.get('decisionGraph').dispatch('keydown', {key: 'ArrowRight'});
  inspected = JSON.parse(h.elements.get('inspectPayload').textContent);
  assert.equal(inspected.id, 'physics');
  assert.equal(inspected.dependency_inputs.telemetry.player.x, 580);
});

test('pause restart and objective handlers send the intended commands', async () => {
  const h = createHarness(); h.update(fixture());
  await h.elements.get('pauseButton').dispatch('click');
  await h.elements.get('restartButton').dispatch('click');
  h.elements.get('modeSelect').value = 'collector';
  await h.elements.get('modeSelect').dispatch('change');
  const commands = h.requests.filter(request => request.url === '/api/command').map(request => JSON.parse(request.options.body));
  assert.deepEqual(commands, [{command: 'pause'}, {command: 'restart'}, {command: 'mode', mode: 'collector'}]);
  h.update({...fixture(), status: 'paused'});
  await h.elements.get('pauseButton').dispatch('click');
  assert.equal(JSON.parse(h.requests.at(-1).options.body).command, 'resume');
});

test('knowledge retention selector sends an explicit opt-in command', async () => {
  const h=createHarness();h.update(fixture());
  h.elements.get('knowledgeScope').value='session';
  await h.elements.get('knowledgeScope').dispatch('change');
  const command=h.requests.find(request=>request.url==='/api/command');
  assert.deepEqual(JSON.parse(command.options.body),{command:'knowledge_scope',scope:'session'});
  assert.equal(h.elements.get('knowledgeScope').value,'run'); // rejected by harness mock
  h.update({...fixture(),knowledge_scope:'session'});
  assert.equal(h.elements.get('knowledgeScope').value,'session');
});

test('graph zoom fit and drag remain interactive without rerunning inference', async () => {
  const h = createHarness(); h.update(fixture());
  await h.elements.get('graphZoomIn').dispatch('click');
  assert(h.inspect('graphView.zoom') > 1);
  await h.elements.get('decisionGraph').dispatch('pointerdown', {clientX: 100, clientY: 100});
  await h.elements.get('decisionGraph').dispatch('pointermove', {clientX: 125, clientY: 130});
  await h.elements.get('decisionGraph').dispatch('pointerup', {clientX: 125, clientY: 130});
  assert.equal(h.inspect('graphView.panX'), 25);
  assert.equal(h.inspect('graphView.panY'), 30);
  await h.elements.get('graphFit').dispatch('click');
  assert.equal(h.inspect('graphView.zoom'), 1);
  assert.equal(h.inspect('graphView.panX'), 0);
  h.animationFrames[0]();
  assert.equal(h.requests.length, 1); // initial GET only
});

test('boot empty graph and terminal graph provenance render without exceptions', () => {
  const h = createHarness();
  h.update({status: 'booting'});
  assert(h.texts.some(text => text.includes('Waiting for executed')));
  h.update({...fixture(), status: 'ended', frame_index: 121, graph_current: false});
  assert(h.texts.some(text => text.includes('LAST PRE-TERMINAL DECISION')));
});

test('fresh pointer tap selects a node without requiring mouse hover', async () => {
  const h = createHarness(); h.update(fixture());
  const point = JSON.parse(h.inspect(`JSON.stringify((()=>{
    const n=graphHitAreas.find(n=>n.key==='controller');
    return {clientX:graphTransform.x+(n.x+n.w/2)*graphTransform.scale,
            clientY:graphTransform.y+(n.y+n.h/2)*graphTransform.scale};
  })())`));
  await h.elements.get('decisionGraph').dispatch('pointerdown', point);
  await h.elements.get('decisionGraph').dispatch('pointerup', point);
  assert.equal(h.elements.get('inspectTitle').textContent, 'CONTROLLER');
});

test('situation report preserves model weights and their source instead of outcome win rate', () => {
  const h = createHarness(), payload = fixture(); h.update(payload);
  assert.equal(h.elements.get('gpuDevice').textContent, 'NVIDIA GeForce RTX 4090 · cuda:0');
  const summary = h.elements.get('situationSummary').textContent;
  assert.match(summary, /STOMP ENEMY 70\.0% · FINISH FAST 15\.0% · COLLECT POWERUP 10\.0%/);
  assert.match(summary, /Controller: RIGHT RUN JUMP/);
  assert.match(summary, /Model changed motor action: no/);
  assert.match(summary, /objective changed motor action: no/);
  const report = JSON.parse(h.elements.get('situationReport').textContent);
  assert.equal(report.model.selected, 'stomp_enemy');
  assert.equal(report.model.goal_scores.stomp_enemy, .7);
  assert.equal(report.model.model_id, 'Open-JEV-2B-fixture');
  assert.equal(report.model.cuda_device, 'cuda:0');
  assert.equal(report.snapshot.model_source, '3:90');
  assert.equal(report.snapshot.frame, 120);
  assert.equal(report.snapshot.model_age_frames, 30);
  assert.equal(report.controller.strategy_applied, false);
  assert.equal(report.controller.intent_accepted, false);
  assert.deepEqual(report.learned_context, payload.knowledge.context);
  assert.match(report.limitations, /not neural weight training/);
  assert.match(report.limitations, /not win probabilities/);
  const weights = h.elements.get('probabilities').innerHTML;
  payload.knowledge.outcomes = {successes: 99, failures: 0};
  payload.episode_reward = 99999;
  h.update(payload);
  assert.equal(h.elements.get('probabilities').innerHTML, weights, 'script outcomes must not change displayed model weights');
});

test('experience map renders current remembered and empty observed columns only for current stage', () => {
  const h = createHarness(); h.update(fixture());
  assert.equal(h.elements.get('knowledgeMap').width, 2400);
  assert.equal(h.elements.get('knowledgeMap').height, 440);
  assert.equal(h.elements.get('knowledgeCount').textContent, '3 COLUMNS · 4 JUMPS');
  assert.equal(h.rectangles.filter(rect => rect.color === '#358957').length, 2);
  assert.equal(h.rectangles.filter(rect => rect.color === '#244131').length, 2);
  assert.equal(h.rectangles.filter(rect => rect.color === '#498799').length, 1);
  assert.equal(h.rectangles.filter(rect => rect.color === '#fa7878').length, 1);
  assert(h.rectangles.every(rect => [rect.x, rect.y, rect.width, rect.height].every(Number.isFinite)));
  const summary = h.elements.get('knowledgeSummary').textContent;
  assert.match(summary, /mean air time 36\.0 frames, range 113\.0 px/);
  assert.match(summary, /Recorded clears\/deaths: 9\/1/);
  assert.match(summary, /neural weights are unchanged/);
  const report = JSON.parse(h.elements.get('knowledgeGraph').textContent);
  assert.equal(report.context.control_adaptation, false);
  assert.equal(report.enemy_types[0].velocity_x_px_per_frame.mean, -1);
  assert(report.graph.edges.some(edge => edge.from === 'enemy_type_6' && edge.to === 'memory_context'));
  const compact = h.sandbox.inspectorValue('compact_state', fixture());
  assert.deepEqual(JSON.parse(JSON.stringify(compact.dependency_inputs.experience_memory)), fixture().knowledge.context);
});

test('missing inference and unverified runtime do not fabricate model judgment or verified GPU', () => {
  const h = createHarness(), payload = fixture();
  payload.decision = {source: 'jevtpp_reactive_controller', reason: 'gap'};
  payload.runtime.validated = false;
  h.update(payload);
  assert.equal(h.elements.get('gpuDevice').textContent, 'DEVICE NOT VERIFIED');
  assert.match(h.elements.get('situationSummary').textContent, /No model judgment yet/);
  const report = JSON.parse(h.elements.get('situationReport').textContent);
  assert.equal(report.model.selected, null);
  assert.equal(report.model.cuda_device, null);
  assert.deepEqual(report.model.goal_scores, {});
  assert.equal(report.snapshot.model_source, null);
  assert.equal(h.elements.get('activeGoal').textContent, 'WAITING FOR MODEL → REFLEX');
});

test('high RAM confidence displays live grid instead of stale remembered ground', () => {
  const h = createHarness(), payload = fixture();
  payload.state.terrain.observation_reliability = 'high_ram';
  payload.debug.trusted_grid = ['STALE'];
  h.update(payload);
  assert.equal(h.elements.get('gridSource').textContent, 'LIVE RAM');
  assert.equal(h.elements.get('tileGrid').textContent, payload.debug.local_grid.join('\n'));
  payload.state.terrain.observation_reliability = 'low_airborne';
  h.update(payload);
  assert.equal(h.elements.get('gridSource').textContent, 'LAST TRUSTED GROUND');
  assert.equal(h.elements.get('tileGrid').textContent, 'STALE');
});

test('digital twin and feasible goal resolution show current forecast without replacing raw model scores', () => {
  const h = createHarness(), payload = fixture();
  payload.decision.raw_model_goal = 'stomp_enemy';
  payload.decision.active_goal = 'collect_powerup';
  payload.decision.intent_accepted = true;
  payload.decision.intent_selection = 'best_feasible_model_score';
  payload.decision.selected_goal_probability = .1;
  h.update(payload);
  assert.match(h.elements.get('twinSummary').textContent, /contact 40\.0 fr/);
  assert.match(h.elements.get('twinProvenance').textContent, /SOURCE CURRENT RAM PROJECTION/);
  assert.match(h.elements.get('twinProvenance').textContent, /ENEMY VELOCITY CURRENT RAM/);
  const report = JSON.parse(h.elements.get('situationReport').textContent);
  assert.equal(report.digital_twin.enemy.projected_distance_px, 128);
  assert.equal(report.goal_resolution.raw_model_goal, 'stomp_enemy');
  assert.equal(report.goal_resolution.selected_feasible_goal, 'collect_powerup');
  assert.equal(report.goal_resolution.selected_score, .1);
  assert.equal(report.model.goal_scores.stomp_enemy, .7);
  assert.equal(h.elements.get('activeGoal').textContent, 'STOMP ENEMY → COLLECT POWERUP');
  payload.digital_twin.enemy.contact_eta_frames = null;
  payload.digital_twin.landing = {status: 'unknown', eta_frames: null, source: 'insufficient_samples', samples: 0};
  h.update(payload);
  assert.match(h.elements.get('twinSummary').textContent, /contact unknown/);
  assert.match(h.elements.get('twinSummary').textContent, /landing UNKNOWN/);
});

test('full game session and incremental nodes display actual lives risk and zero knowledge after GAME OVER', () => {
  const h = createHarness(), payload = fixture();
  payload.version = 'v10-full-game';
  payload.session = {run_id: 2, world: 2, stage: 1, lives_remaining: 1, stages_cleared: 4,
    deaths: 2, memory_generation: 2, game_over: false, game_complete: false};
  payload.game_phase = {phase: 'playing', lives_remaining: 1};
  payload.learning_delta = {source: 'current_observation_delta', new_columns: 2, new_enemy_types: 1,
    enemy_velocity_samples: 0, jump_samples: 0};
  payload.risk_policy = {stance: 'cautious', allow_aggressive_exploration: false, reasons: ['last_life', 'new_enemy_type']};
  const added = {
    game_phase: ['telemetry'], learning_delta: ['experience_memory'],
    risk_policy: ['game_phase', 'digital_twin', 'learning_delta'],
  };
  for (const [id, dependencies] of Object.entries(added)) {
    payload.graph.nodes.push({id, dependencies, output: payload[id], status: 'succeeded', snapshot_id: '3:120', duration_ms: .01});
    payload.graph.edges.push(...dependencies.map(from => ({from, to: id, active: true, bytes: 24})));
  }
  for (const id of ['compact_state', 'controller']) {
    const item = payload.graph.nodes.find(node => node.id === id);
    for (const from of Object.keys(added)) {
      item.dependencies.push(from);
      payload.graph.edges.push({from, to: id, active: true, bytes: 24});
    }
  }
  h.update(payload);
  assert.equal(h.elements.get('worldStage').textContent, '2-1');
  assert.equal(h.elements.get('lives').textContent, 1);
  assert.equal(h.elements.get('progressBar').style.width, '12.5%');
  assert.equal(h.elements.get('sessionPhase').textContent, 'PLAYING');
  assert.match(h.elements.get('situationSummary').textContent, /risk CAUTIOUS/);
  const report = JSON.parse(h.elements.get('situationReport').textContent);
  assert.equal(report.learning_delta.new_columns, 2);
  assert.equal(report.risk_policy.allow_aggressive_exploration, false);
  const inspected = h.sandbox.inspectorValue('controller', payload);
  assert.equal(inspected.dependency_inputs.risk_policy.stance, 'cautious');
  assert.equal(inspected.dependency_inputs.learning_delta.new_enemy_types, 1);
  payload.status = 'game_over'; payload.game_phase.phase = 'game_over';
  payload.session.game_over = true; payload.session.lives_remaining = 0; payload.session.memory_generation = 3;
  payload.knowledge = {columns: [], enemy_types: [], observed_frames: 0};
  payload.graph_current = false;
  h.update(payload);
  assert.equal(h.elements.get('lives').textContent, 0);
  assert.equal(h.elements.get('sessionPhase').textContent, 'GAME OVER');
  assert(h.elements.get('sessionPhase').classList.values.has('terminal'));
  assert.equal(h.elements.get('knowledgeCount').textContent, '0 COLUMNS · 0 JUMPS');
  assert.equal(h.elements.get('memoryGeneration').textContent, 3);
});

test('flight feedback and hunter authority are visible without model attribution', () => {
  const h=createHarness(),payload=fixture();
  for(const [id,output] of Object.entries({
    action_dynamics:{samples:24,jump_cut:true,last_flight:{signature:'traverse:right:28:0',result:'life_lost_before_verified_landing'}},
    hunter_engagement:{apply:true,phase:'track_intercept'},
    persistent_tasks:{clearance:'incomplete',confirmed_eliminations:2,unresolved_targets:1}
  })) payload.graph.nodes.push({id,dependencies:['telemetry'],output,status:'succeeded',duration_ms:.01});
  payload.decision.arbitration={selected:'hunter',proposals:[]};
  h.update(payload);
  assert.equal(h.elements.get('learnerStatus').textContent,'HUNTER CONTROLS');
  assert.match(h.elements.get('taskSummary').textContent,/2 confirmed, 1 unresolved/);
  assert.match(h.elements.get('taskSummary').textContent,/CUT \(cannot restore height\)/);
  const report=JSON.parse(h.elements.get('situationReport').textContent);
  assert.equal(report.online_learning.hunter_engagement.phase,'track_intercept');
  assert.equal(report.online_learning.action_dynamics.last_flight.signature,'traverse:right:28:0');
});

test('model selector uses real availability and LAYA provenance', async () => {
  const h=createHarness(),payload=fixture();
  payload.model_selection={selected:'laya',options:[{id:'open_jev',label:'Open-JEV',available:true,reason:'Ready'},{id:'laya',label:'LAYA',available:true,reason:'Ready'}]};
  payload.runtime={validated:true,device:'cuda:0',device_name:'RTX 4090',model_kind:'laya',base_model_id:'laya-multilingual'};
  h.update(payload);
  assert.equal(h.elements.get('modelSelect').value,'laya');
  assert.match(h.elements.get('modelStack').textContent,/LAYA/);
  assert.doesNotMatch(h.elements.get('modelStack').textContent,/OPEN-JEV/);
  h.elements.get('modelSelect').value='open_jev';
  await h.elements.get('modelSelect').dispatch('change');
  assert(h.requests.some(r=>r.options?.body&&JSON.parse(r.options.body).command==='model'));
});

test('plan scores and single final authority are distinct from legacy goals', () => {
  const h=createHarness(),payload=fixture();
  payload.version='v14-generic-skills';
  Object.assign(payload.decision,{judgment_kind:'plan',model_plan:'transfer:right',plan_scores:{'transfer:right':.7},candidate_labels:{'transfer:right':'transfer: right_jump'},defer_score:.3,
    plan_intent_accepted:true,plan_model_applied:true,plan_intent:{accepted:true,reason:'current_context_and_candidate_verified'},arbitration:{selected:'planner',proposals:[]}});
  h.update(payload);
  assert.match(h.elements.get('confidence').textContent,/TOP OPTION 70/);
  assert.match(h.elements.get('dashboardVersion').textContent,/v14/);
  assert.match(h.elements.get('gateReason').textContent,/ARBITER planner/);
  const report=JSON.parse(h.elements.get('situationReport').textContent);
  assert.equal(report.model.kind,'plan');assert.equal(report.model.goal_scores,null);
  assert.equal(report.plan_judgment.scores['transfer:right'],.7);
  assert.equal(report.arbitration.selected,'planner');
});

(async () => {
  let failures = 0;
  for (const {name, body} of tests) {
    try { await body(); console.log(`PASS ${name}`); }
    catch (error) { failures++; console.error(`FAIL ${name}\n${error.stack}`); }
  }
  console.log(`${tests.length - failures}/${tests.length} dashboard contract tests passed (not visual QA)`);
  process.exitCode = failures ? 1 : 0;
})();
