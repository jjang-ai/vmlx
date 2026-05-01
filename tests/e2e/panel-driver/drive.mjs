/* Playwright-Electron E2E driver for the vMLX panel.
 *
 * Why custom: the Electron renderer is opaque to System Events
 * (every SwiftUI/Chromium widget reports as AXGroup), so AppleScript
 * can't drive it. Playwright connects over the Chrome DevTools Protocol
 * (CDP) port that Electron exposes when launched with
 * --remote-debugging-port=<port>, then walks the renderer DOM the same
 * way it walks Chrome — full querySelector + click + type + screenshot
 * + console capture.
 *
 * Usage:
 *   node drive.mjs <subcommand> [args...]
 *
 *   launch                  — launch /Applications/vMLX.app with CDP on 9222
 *   smoke                   — full smoke-test pass (setup-screen check + tab nav + screenshot)
 *   shot <out.png>          — screenshot the visible renderer window
 *   click <selector>        — click a DOM element by CSS selector
 *   type  <selector> <text> — type into an input
 *   eval  <js-expr>         — evaluate JS expression in the renderer, print result
 *   wait  <selector> [ms]   — wait for selector to appear (default 10000)
 *   logs                    — print captured console logs and exit
 *   close                   — quit the running panel
 *
 * The script is fully self-contained — no shared state across invocations.
 */
import { chromium } from 'playwright';
import { spawn } from 'child_process';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPORTS = resolve(HERE, '..', 'reports');
mkdirSync(REPORTS, { recursive: true });

const APP_PATH = '/Applications/vMLX.app/Contents/MacOS/vMLX';
const CDP_PORT = Number(process.env.VMLX_CDP_PORT || 9222);
const CDP_URL = `http://127.0.0.1:${CDP_PORT}`;

function logTs(msg) {
  process.stderr.write(`[panel-driver ${new Date().toISOString()}] ${msg}\n`);
}

async function connectCDP() {
  return await chromium.connectOverCDP(CDP_URL, { timeout: 15000 });
}

async function getRenderer(browser) {
  const ctxs = browser.contexts();
  for (const ctx of ctxs) {
    for (const page of ctx.pages()) {
      const url = page.url();
      // Renderer is loaded from the asar, file://… or http://localhost:5173 (dev)
      if (url.startsWith('file://') || url.includes('localhost') || url.includes('app.asar')) {
        return page;
      }
    }
  }
  // fall back to first page
  return ctxs[0]?.pages()?.[0];
}

function launch() {
  if (!existsSync(APP_PATH)) {
    logTs(`ERR: ${APP_PATH} not found`);
    process.exit(2);
  }
  logTs(`launching with --remote-debugging-port=${CDP_PORT}`);
  const child = spawn(APP_PATH, [`--remote-debugging-port=${CDP_PORT}`, '--no-sandbox'], {
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', d => process.stdout.write(`[panel] ${d}`));
  child.stderr.on('data', d => process.stderr.write(`[panel] ${d}`));
  child.unref();
  logTs(`pid=${child.pid}`);
}

async function shot(outPath) {
  const browser = await connectCDP();
  try {
    const page = await getRenderer(browser);
    if (!page) throw new Error('no renderer page found');
    const out = resolve(outPath);
    await page.screenshot({ path: out, fullPage: false });
    logTs(`wrote ${out}`);
    console.log(out);
  } finally { await browser.close(); }
}

async function click(selector) {
  const browser = await connectCDP();
  try {
    const page = await getRenderer(browser);
    await page.waitForSelector(selector, { timeout: 10000 });
    await page.click(selector);
    logTs(`clicked ${selector}`);
  } finally { await browser.close(); }
}

async function type_(selector, text) {
  const browser = await connectCDP();
  try {
    const page = await getRenderer(browser);
    await page.waitForSelector(selector, { timeout: 10000 });
    await page.fill(selector, text);
    logTs(`typed into ${selector}`);
  } finally { await browser.close(); }
}

async function evalJs(expr) {
  const browser = await connectCDP();
  try {
    const page = await getRenderer(browser);
    const result = await page.evaluate(expr);
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
}

async function wait_(selector, timeoutMs = 10000) {
  const browser = await connectCDP();
  try {
    const page = await getRenderer(browser);
    await page.waitForSelector(selector, { timeout: Number(timeoutMs) });
    logTs(`saw ${selector}`);
  } finally { await browser.close(); }
}

async function close_() {
  const browser = await connectCDP();
  try {
    const ctx = browser.contexts()[0];
    for (const p of ctx.pages()) await p.close();
  } finally { await browser.close(); }
}

async function smoke() {
  const report = { ts: new Date().toISOString(), tests: [] };
  function add(name, ok, details = '') {
    report.tests.push({ name, ok, details });
    console.log(`  ${ok ? 'OK ' : 'FAIL'}  ${name}  ${details}`);
  }

  const browser = await connectCDP();
  const page = await getRenderer(browser);
  if (!page) { console.log('no renderer page'); process.exit(2); }

  // Capture console + errors
  const consoleLines = [];
  page.on('console', m => consoleLines.push(`${m.type()}: ${m.text()}`));
  page.on('pageerror', e => consoleLines.push(`error: ${e.message}`));

  try {
    // 1. URL + title
    add('renderer-url', !!page.url(), page.url());
    add('renderer-title', !!(await page.title()), await page.title());

    // 2. setup-screen detection — fail if "Install Engine" button visible
    const setupVisible = await page.evaluate(() => {
      const txt = document.body.innerText || '';
      return /First-time setup|Install Engine/i.test(txt);
    });
    add('no-setup-screen', !setupVisible, setupVisible ? 'STILL SEEING SETUP SCREEN' : 'engine detected');

    // 3. tab navigation — Chat / Server / Tools / Image / API
    const tabs = await page.evaluate(() => {
      const all = [...document.querySelectorAll('button, [role="tab"], a')]
        .map(e => (e.innerText || '').trim())
        .filter(t => /^(Code|Chat|Server|Tools|Image|API)$/.test(t));
      return [...new Set(all)];
    });
    add('tab-set-present', tabs.length >= 4, tabs.join(','));

    // 4. screenshot baseline
    const shotPath = resolve(REPORTS, `smoke-${Date.now()}.png`);
    await page.screenshot({ path: shotPath, fullPage: false });
    add('screenshot', existsSync(shotPath), shotPath);

    // 5. console error count
    const errorCount = consoleLines.filter(l => l.startsWith('error:')).length;
    add('no-console-errors', errorCount === 0, `count=${errorCount}`);

    // 6. engine version surface (if present in DOM)
    const engineInfo = await page.evaluate(() => {
      const m = (document.body.innerText || '').match(/vmlx[_\s-]?engine[^\d]*(\d+\.\d+\.\d+)/i);
      return m ? m[1] : null;
    });
    add('engine-version-surfaced', !!engineInfo, engineInfo || '(not surfaced in current view)');

    writeFileSync(resolve(REPORTS, `smoke-${Date.now()}.json`), JSON.stringify({ ...report, console: consoleLines }, null, 2));
    console.log(`\n${report.tests.filter(t => t.ok).length}/${report.tests.length} passed`);
    process.exit(report.tests.every(t => t.ok) ? 0 : 1);
  } finally {
    await browser.close();
  }
}

async function logs() {
  const browser = await connectCDP();
  try {
    const page = await getRenderer(browser);
    const lines = await page.evaluate(() => {
      // Pull anything stashed on window.__vmlxLogs (test-only hook); otherwise
      // we can only pull future console events. Drive `logs` BEFORE running
      // any other action and then again after to see the diff.
      return (window).__vmlxLogs || [];
    });
    console.log(JSON.stringify(lines, null, 2));
  } finally { await browser.close(); }
}

const cmd = process.argv[2];
const args = process.argv.slice(3);
try {
  switch (cmd) {
    case 'launch': launch(); break;
    case 'shot':   await shot(args[0] || resolve(REPORTS, `shot-${Date.now()}.png`)); break;
    case 'click':  await click(args[0]); break;
    case 'type':   await type_(args[0], args[1]); break;
    case 'eval':   await evalJs(args[0]); break;
    case 'wait':   await wait_(args[0], args[1]); break;
    case 'logs':   await logs(); break;
    case 'close':  await close_(); break;
    case 'smoke':  await smoke(); break;
    default:
      console.error('usage: drive.mjs {launch|shot|click|type|eval|wait|logs|close|smoke}');
      process.exit(1);
  }
} catch (e) {
  console.error(`panel-driver error: ${e.message}`);
  process.exit(1);
}
