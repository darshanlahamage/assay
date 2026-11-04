// Loads the dashboard in jsdom (a browser emulator), opens every run, its first rollout
// and the New-run dialog, and fails on any page script error. Catches what `node --check`
// cannot, such as names that clash with browser globals.
//   node tools/ui_check.js [port]        (server must be running: assay ui --no-browser)
const { JSDOM, VirtualConsole } = require('jsdom');
const base = 'http://127.0.0.1:' + (process.argv[2] || 8765);
const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push(e.message));
vc.on('error', e => errors.push(String(e)));
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const dom = await JSDOM.fromURL(base + '/', { runScripts: 'dangerously', virtualConsole: vc, pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = (u, o) => fetch(new URL(u, base), o);
      if (w.HTMLDialogElement) w.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
    } });
  const w = dom.window, d = w.document;
  await sleep(1200);
  const runs = [...d.querySelectorAll('.run')].map(e => e.dataset.id);
  let rollouts = 0;
  for (const id of runs.slice(0, 15)) {
    await w.eval(`openRun(${JSON.stringify(id)})`); await sleep(400);
    const row = d.querySelector('.row');
    if (row) { row.click(); await sleep(500); rollouts++; }
  }
  d.querySelector('#newBtn').click(); await sleep(600);
  const dialog = d.querySelector('#dlg').open;
  console.log(`runs=${runs.length} rollouts_opened=${rollouts} dialog_open=${dialog} errors=${errors.length}`);
  errors.forEach(e => console.log('ERROR', e));
  process.exit(errors.length || !dialog ? 1 : 0);
})().catch(e => { console.log('ERROR', e.message); process.exit(1); });
