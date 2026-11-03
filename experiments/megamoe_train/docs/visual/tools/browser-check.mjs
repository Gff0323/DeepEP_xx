// Copyright (c) 2026, DeepEP contributors.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const visual = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const artifacts = path.join(visual, '.artifacts');
fs.mkdirSync(artifacts, { recursive: true });
const edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const executablePath = process.env.BROWSER_PATH || (process.platform === 'win32' && fs.existsSync(edge) ? edge : undefined);
const browser = await chromium.launch({ headless: true, executablePath });
const errors = [];
const reports = [];
const server = http.createServer((req, res) => {
  const relative = new URL(req.url, 'http://localhost').pathname;
  if (relative === '/favicon.ico') { res.writeHead(204).end(); return; }
  const file = path.resolve(visual, '.' + (relative === '/' ? '/index.html' : decodeURIComponent(relative)));
  if (!file.startsWith(visual + path.sep)) { res.writeHead(403).end(); return; }
  try { res.setHeader('Content-Type', file.endsWith('.svg') ? 'image/svg+xml' : 'text/html; charset=utf-8'); res.end(fs.readFileSync(file)); }
  catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
async function chooseView(page, id) {
  const button = page.locator(`[data-view="${id}"]`);
  if (!await button.isVisible()) await page.locator('#nav-toggle').click();
  await button.click();
}
try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1060 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(() => window.__reader?.data.graphs.length === 8);
  const ids = await page.evaluate(() => window.__reader.data.graphs.map(g => g.id));
  const syntaxPage = await context.newPage();
  await syntaxPage.addScriptTag({ path: path.join(visual, 'node_modules', 'mermaid', 'dist', 'mermaid.min.js') });
  const diagrams = await page.evaluate(() => window.__reader.data.graphs.map(g => ({ id: g.id, source: g.mermaid })));
  for (const diagram of diagrams) await syntaxPage.evaluate(async diagram => {
    window.mermaid.initialize({ startOnLoad: false });
    await window.mermaid.parse(diagram.source);
  }, diagram);
  await syntaxPage.close();
  assert.equal(await page.locator('[data-view]').count(), 11);
  assert.equal(await page.locator('.source-location').first().inputValue(), await page.evaluate(() => {
    const source = window.__reader.selected().sources[0]; return `${source.path}:${source.line}`;
  }));
  for (const id of ids) {
    await chooseView(page, id);
    assert.equal(await page.evaluate(() => window.__reader.graph().id), id);
    const first = await page.locator('#canvas [data-node]').first().getAttribute('data-node');
    await page.locator(`#canvas [data-node="${first}"]`).click();
    assert(await page.locator('.code-line.anchor-line').count() > 0, `${id}: source excerpt`);
    assert((await page.locator('.source-links a').first().getAttribute('href')).includes('/blob/'));
    const nodes = await page.locator('#canvas [data-node]').count();
    await page.locator('#next').click();
    assert.equal(await page.evaluate(() => window.__reader.selected().ordinal), 2);
    await page.locator('#previous').click();
    assert.equal(await page.evaluate(() => window.__reader.selected().ordinal), 1);
    await page.locator('#query').fill('不存在的函数xyz');
    assert.equal(await page.locator('#canvas .node.muted').count(), nodes);
    await page.locator('#query').fill('');
    await page.locator('#show-sync').uncheck();
    assert(await page.locator('#viewport').evaluate(el => el.classList.contains('hide-sync')));
    await page.locator('#show-sync').check();
    const before = await page.locator('#zoom-label').innerText();
    await page.locator('#zoom-in').click();
    assert.notEqual(await page.locator('#zoom-label').innerText(), before);
    await page.locator('#fit').click();
    await page.locator('#expand').click();
    assert.equal(await page.locator('#expand').getAttribute('aria-pressed'), 'true');
    await page.locator('#expand').click();
    const clipped = await page.locator('#canvas svg').evaluate(svg => [...svg.querySelectorAll('text')].filter(el => {
      const box = el.getBBox(), view = svg.viewBox.baseVal;
      return box.x < view.x - 2 || box.y < view.y - 2 || box.x + box.width > view.width + 2 || box.y + box.height > view.height + 2;
    }).map(el => el.textContent));
    assert.deepEqual(clipped, [], `${id}: clipped text`);
    const overflowing = await page.locator('#canvas .node').evaluateAll(nodes => nodes.flatMap(node => {
      const rect = node.querySelector('.node-box');
      const x = +rect.getAttribute('x'), y = +rect.getAttribute('y');
      const width = +rect.getAttribute('width'), height = +rect.getAttribute('height');
      return [...node.querySelectorAll('text')].filter(text => {
        const box = text.getBBox();
        return box.x < x || box.y < y || box.x + box.width > x + width || box.y + box.height > y + height;
      }).map(text => `${node.dataset.node}: ${text.textContent}`);
    }));
    assert.deepEqual(overflowing, [], `${id}: text exceeds its node`);
    await page.screenshot({ path: path.join(artifacts, `${id}.png`) });
    reports.push({ id, nodes, source: true, navigation: true, clipped });
    // Static PNGs use the same generated SVG as the interactive reader.
    const svgPage = await context.newPage();
    await svgPage.goto(pathToFileURL(path.join(visual, 'diagrams', `${id}.svg`)).href);
    const image = path.join(visual, 'diagrams', `${id}.png`);
    await svgPage.locator('svg').screenshot({ path: image, scale: 'css' });
    fs.copyFileSync(image, path.join(visual, 'site', 'diagrams', `${id}.png`));
    await svgPage.close();
  }
  const scenes = await page.evaluate(() => window.__reader.data.learning.scenes.map(scene => scene.id));
  const focus = await page.evaluate(() => window.__reader.data.learning.case.focusTokens.map(token => token.id));
  const sceneReports = [];
  for (const id of scenes) {
    await chooseView(page, id);
    assert.equal(await page.evaluate(() => window.__reader.view()), id);
    assert(await page.locator('#canvas').evaluate(canvas => canvas.clientWidth <= canvas.parentElement.clientWidth + 2), `${id}: content fits its panel`);
    await page.locator('#case-token').selectOption(focus[0]);
    if (id === 'learn-token') await page.locator('#case-phase').selectOption('forward');
    assert(await page.locator('#canvas [data-learning-node]:visible').count() > 0, `${id}: clickable sources`);
    const source = page.locator('#canvas [data-learning-node]:visible').first();
    const nodeId = await source.getAttribute('data-learning-node');
    await source.click();
    assert.equal(await page.evaluate(() => window.__reader.selected().id), nodeId);
    assert.equal(await page.evaluate(() => window.__reader.view()), id, 'Source inspection keeps the scene');
    const output = path.join(visual, 'diagrams', `${id}.png`);
    await page.locator('#viewport').screenshot({ path: output, scale: 'css' });
    fs.copyFileSync(output, path.join(visual, 'site', 'diagrams', `${id}.png`));
    await page.locator('#case-token').selectOption(focus[1]);
    if (id === 'learn-token') await page.locator('#case-phase').selectOption('backward');
    if (id === 'learn-saved') {
      await page.locator('#canvas [data-learning-resource]').first().click();
      assert.equal(await page.locator('#canvas .learning-resource-active').count(), 1, 'Producer and consumers share the selected tensor');
    }
    const before = await page.evaluate(() => window.__reader.learningState());
    await chooseView(page, '00-overview');
    await chooseView(page, id);
    assert.deepEqual(await page.evaluate(() => window.__reader.learningState()), before, `${id}: preserved context`);
    const invalidVisible = await page.locator('#canvas [data-learning-token]:visible').evaluateAll((elements, token) => elements.filter(el => el.dataset.learningToken !== token).map(el => el.dataset.learningToken), focus[1]);
    assert.deepEqual(invalidVisible, [], `${id}: visible token slices`);
    await page.screenshot({ path: path.join(artifacts, `${id}.png`) });
    sceneReports.push({ id, source: true, tokenAndPhase: true, context: true });
  }
  const sceneHash = await page.evaluate(() => location.hash);
  await page.reload();
  assert.equal(await page.evaluate(() => location.hash), sceneHash, 'Scene deep link reload');
  await chooseView(page, '06-backward-kernel');
  const download = page.waitForEvent('download');
  await page.locator('#mmd-download').click();
  assert.equal((await download).suggestedFilename(), '06-backward-kernel.mmd');
  await page.screenshot({ path: path.join(artifacts, 'desktop.png') });
  // Verify that navigation down to another view actually selects that view.
  const targets = await page.evaluate(() => window.__reader.data.graphs.flatMap(graph => graph.nodes
    .filter(node => node.drilldown).map(node => ({ sourceView: graph.id, sourceNode: node.id, ...node.drilldown }))));
  for (const target of targets) {
    await page.evaluate(target => { window.__reader.showView(target.sourceView, target.sourceNode); }, target);
    await page.locator('[data-drill-view]').click();
    assert.equal(await page.evaluate(() => window.__reader.graph().id), target.view);
    if (target.node) assert.equal(await page.evaluate(() => window.__reader.selected().id), target.node);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await chooseView(page, 'learn-token');
  assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), 'Mobile page overflow');
  await page.screenshot({ path: path.join(artifacts, 'mobile.png'), fullPage: true });
  await chooseView(page, '00-overview');
  // A fresh context with all network requests blocked must still read file://.
  const offline = await browser.newContext({ viewport: { width: 1440, height: 960 }, offline: true });
  const local = await offline.newPage();
  local.on('pageerror', error => errors.push(error.message));
  await local.goto(pathToFileURL(path.join(visual, 'index.html')).href);
  await local.waitForFunction(() => window.__reader?.data.graphs.length === 8);
  const localSources = await local.evaluate(() => [...new Set(window.__reader.data.graphs
    .flatMap(graph => graph.nodes.flatMap(node => node.sources.map(source => source.path))))]
    .map(file => new URL(`../../../../${file}`, location.href).href));
  for (const source of localSources) assert(fs.existsSync(fileURLToPath(source)), `Local source file: ${source}`);
  await chooseView(local, '06-backward-kernel');
  assert.equal(await local.locator('.source-links a[href^="file:"]').count(), 0, 'No misleading raw file link');
  const source = await local.evaluate(() => window.__reader.selected().sources[0]);
  const sourcePath = path.resolve(visual, '../../../..', source.path);
  const location = await local.locator('.source-location').first().inputValue();
  assert.equal(location.toLowerCase(), `${sourcePath.replaceAll('\\', '/')}:${source.line}`.toLowerCase());
  const editor = new URL(await local.locator('.source-links a[href^="vscode:"]').first().getAttribute('href'));
  assert.equal(editor.host, 'file');
  assert.equal(editor.hash + editor.search, '', 'Editor path must not become a query or fragment');
  assert.equal(decodeURIComponent(editor.pathname).replace(/^\/([A-Za-z]:\/)/, '$1').toLowerCase(), `${location}:1`.toLowerCase());
  await local.locator('[data-copy-location]').first().click();
  const copyStatus = await local.locator('.copy-status').first().innerText();
  assert(copyStatus.startsWith('已复制') || copyStatus.startsWith('路径已选中'), 'Copy or explicit keyboard fallback');
  const hash = await local.evaluate(() => location.hash);
  await local.reload();
  assert.equal(await local.evaluate(() => location.hash), hash);
  await local.screenshot({ path: path.join(artifacts, 'offline.png') });
  await offline.close();
  // The optional editor host uses native messages instead of file/protocol navigation.
  const hosted = await browser.newContext({ viewport: { width: 1440, height: 960 } });
  await hosted.addInitScript(() => {
    window.__editorMessages = []; window.__editorAcquisitions = 0;
    window.acquireVsCodeApi = () => {
      ++window.__editorAcquisitions;
      return { postMessage: message => window.__editorMessages.push(message) };
    };
  });
  const native = await hosted.newPage();
  native.on('pageerror', error => errors.push(error.message));
  await native.goto(`http://127.0.0.1:${server.address().port}`);
  assert.equal(await native.evaluate(() => window.__editorAcquisitions), 1);
  assert.equal(await native.locator('[data-copy-location],.source-location').count(), 0);
  await chooseView(native, 'learn-token');
  const teachingTarget = native.locator('#canvas [data-learning-node]:visible').first();
  const teachingId = await teachingTarget.getAttribute('data-learning-node');
  await teachingTarget.click();
  assert.deepEqual(await native.evaluate(() => window.__editorMessages.at(-1)), { type: 'openSource', nodeId: teachingId, sourceIndex: 0 });
  await chooseView(native, '00-overview');
  const selectedId = await native.evaluate(() => window.__reader.selected().id);
  await native.locator('[data-open-source]').first().click();
  assert.deepEqual(await native.evaluate(() => window.__editorMessages.at(-1)), { type: 'openSource', nodeId: selectedId, sourceIndex: 0 });
  const graphNode = await native.locator('#canvas [data-node]').nth(1).getAttribute('data-node');
  await native.locator(`#canvas [data-node="${graphNode}"]`).click();
  assert.deepEqual(await native.evaluate(() => window.__editorMessages.at(-1)), { type: 'openSource', nodeId: graphNode, sourceIndex: 0 });
  await native.locator('#source-map').click();
  assert.deepEqual(await native.evaluate(() => window.__editorMessages.at(-1)), { type: 'openMapping' });
  await native.screenshot({ path: path.join(artifacts, 'editor-bridge.png') });
  await hosted.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(artifacts, 'browser-check.json'), JSON.stringify({ ok: true, errors, views: reports, scenes: sceneReports, offline: true, mobile: true, editorBridge: true, drilldown: targets.length, downloads: true }, null, 2) + '\n');
  console.log(`Browser PASS: ${reports.length} source views + ${sceneReports.length} learning scenes; offline, mobile, source cards, context and native bridge.`);
} catch (error) {
  fs.writeFileSync(path.join(artifacts, 'browser-error.txt'), String(error.stack));
  throw error;
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
