// Copyright (c) 2026, DeepEP contributors.
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('reader-data').textContent);
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  const groups = [ ['L1', '项目架构'], ['L2', '模块关系'], ['L3', '执行生命周期'], ['L4', '函数调用链'], ['L5', '核内与内存'] ];
  let graph, selected, scale = 1;
  const canvas = $('canvas'), viewport = $('viewport');
  $('repository').href = data.sourceRepository;
  $('revision').href = `${data.sourceRepository}/tree/${data.sourceCommit}`;
  $('revision').textContent = `源码 ${data.sourceCommit.slice(0, 7)}`;
  $('navigation').innerHTML = groups.map(([layer, title]) => `<div class="nav-group"><span class="nav-label">${layer} · ${title}</span>${data.graphs.filter(g => g.layer === layer).map(g => `<button type="button" data-view="${g.id}">${esc(g.title)}</button>`).join('')}</div>`).join('');
  function setScale(value) {
    scale = Math.max(.55, Math.min(1.75, value));
    const svg = canvas.querySelector('svg');
    svg.style.width = `${graph.width * scale}px`;
    svg.style.height = `${graph.height * scale}px`;
    $('zoom-label').textContent = `${Math.round(scale * 100)}%`;
  }
  function fit() { setScale(Math.min(1, (viewport.clientWidth - 18) / graph.width)); }
  function updateHash() { history.replaceState(null, '', `#view=${graph.id}&node=${selected.id}`); }
  function localFile(source) {
    if (location.protocol !== 'file:') return '';
    return new URL(`../../../../${source.path}`, location.href).href;
  }
  function editorFile(source) {
    const local = localFile(source);
    if (!local) return '';
    const file = new URL(local);
    const fullPath = decodeURIComponent(file.pathname).replace(/^\/([A-Za-z]:\/)/, '$1');
    return `vscode://file/${fullPath}:${source.line}`;
  }
  function details(node) {
    const lane = graph.lanes.find(l => l.id === node.lane);
    const list = [['读', node.reads], ['写', node.writes], ['等', node.waits]];
    const sourceBlocks = node.sources.map((source, index) => `<details class="source-block" ${index === 0 ? 'open' : ''}><summary>${esc(source.path.split('/').at(-1))}:${source.line}</summary><div class="source-links"><a target="_blank" rel="noopener" href="${esc(source.url)}">在线源码 ↗</a>${localFile(source) ? `<a href="${esc(localFile(source))}" target="_blank" rel="noopener">本地文件</a><a href="${esc(editorFile(source))}">编辑器定位</a>` : ''}</div><div class="source-path">${esc(source.path)}</div><div class="code-lines">${source.excerpt.map(row => `<div class="code-line ${row.line === source.line ? 'anchor-line' : ''}"><span class="ln">${row.line}</span><code>${esc(row.text)}</code></div>`).join('')}</div></details>`).join('');
    return `<div class="detail-kicker"><span class="pill">${esc(lane.label)}</span><span>${node.ordinal} / ${graph.nodes.length}</span></div><h3>${esc(node.title)}</h3>${node.symbol ? `<code class="function-name">${esc(node.symbol)}</code>` : ''}${(node.description || []).map(p => `<p>${esc(p)}</p>`).join('')}<dl class="detail-list">${list.filter(([, values]) => values?.length).map(([title, values]) => `<div><dt>${title}</dt><dd>${values.map(esc).join('<br>')}</dd></div>`).join('')}</dl>${node.formula ? `<pre class="formula">${esc(node.formula)}</pre>` : ''}${node.drilldown ? `<button type="button" class="drilldown" data-drill-view="${esc(node.drilldown.view)}" data-drill-node="${esc(node.drilldown.node || '')}">继续深入 → ${esc(data.graphs.find(g => g.id === node.drilldown.view).title)}</button>` : ''}<div class="detail-nav"><button type="button" id="previous" ${node.ordinal === 1 ? 'disabled' : ''}>← 上一步</button><button type="button" id="next" ${node.ordinal === graph.nodes.length ? 'disabled' : ''}>下一步 →</button></div><div class="sources-title">对应源码</div>${sourceBlocks}`;
  }
  function select(id, scroll = false) {
    selected = graph.nodes.find(n => n.id === id) || graph.nodes[0];
    for (const node of canvas.querySelectorAll('[data-node]')) node.classList.toggle('selected', node.dataset.node === selected.id);
    for (const edge of canvas.querySelectorAll('.edge')) {
      const related = edge.dataset.from === selected.id || edge.dataset.to === selected.id;
      edge.classList.toggle('related', related);
      edge.classList.toggle('dimmed', !related);
    }
    $('inspector').innerHTML = details(selected);
    $('previous').onclick = () => select(graph.nodes[selected.ordinal - 2]?.id, true);
    $('next').onclick = () => select(graph.nodes[selected.ordinal]?.id, true);
    $('inspector').querySelector('[data-drill-view]')?.addEventListener('click', event => {
      showView(event.currentTarget.dataset.drillView, event.currentTarget.dataset.drillNode);
    });
    if (scroll) {
      const x = (selected.x + selected.w / 2) * scale - viewport.clientWidth / 2;
      const y = selected.y * scale - 65;
      viewport.scrollTo({ left: Math.max(0, x), top: Math.max(0, y), behavior: 'smooth' });
    }
    updateHash();
  }
  function search() {
    const q = $('query').value.trim().toLowerCase();
    let matched = 0;
    for (const node of graph.nodes) {
      const text = JSON.stringify([node.title, node.symbol, node.description, node.reads, node.writes, node.waits]).toLowerCase();
      const visible = !q || text.includes(q);
      canvas.querySelector(`[data-node="${node.id}"]`).classList.toggle('muted', !visible);
      if (visible) ++matched;
    }
    $('search-status').textContent = q ? `找到 ${matched} / ${graph.nodes.length} 个节点` : '点击节点查看说明和源码';
  }
  function showView(id, node) {
    graph = data.graphs.find(g => g.id === id) || data.graphs[0];
    $('view-title').textContent = graph.title;
    $('view-layer').textContent = `${graph.layer} · ${groups.find(([key]) => key === graph.layer)[1]}`;
    $('view-summary').textContent = graph.summary;
    $('node-count').textContent = `${graph.nodes.length} 个节点`;
    for (const button of $('navigation').querySelectorAll('[data-view]')) {
      const active = button.dataset.view === graph.id;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    }
    canvas.innerHTML = graph.svg;
    viewport.scrollTo(0, 0);
    $('query').value = '';
    fit(); search(); select(node);
    for (const element of canvas.querySelectorAll('[data-node]')) {
      element.onclick = () => select(element.dataset.node);
      element.onkeydown = event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); select(element.dataset.node); } };
    }
  }
  function download(content, type, name) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  }
  for (const button of $('navigation').querySelectorAll('[data-view]')) button.onclick = () => showView(button.dataset.view);
  $('fit').onclick = fit;
  $('zoom-in').onclick = () => setScale(scale + .1);
  $('zoom-out').onclick = () => setScale(scale - .1);
  $('query').oninput = search;
  $('show-sync').onchange = () => viewport.classList.toggle('hide-sync', !$('show-sync').checked);
  $('expand').onclick = () => { const expanded = viewport.classList.toggle('expanded'); $('expand').setAttribute('aria-pressed', String(expanded)); $('expand').textContent = expanded ? '收起整图' : '展开整图'; };
  $('svg-download').onclick = () => download(graph.svg, 'image/svg+xml', `${graph.id}.svg`);
  $('mmd-download').onclick = () => download(graph.mermaid, 'text/plain', `${graph.id}.mmd`);
  document.addEventListener('keydown', event => {
    if (event.target.matches('input,textarea')) return;
    if (event.key === 'ArrowRight' && selected.ordinal < graph.nodes.length) { event.preventDefault(); select(graph.nodes[selected.ordinal].id, true); }
    if (event.key === 'ArrowLeft' && selected.ordinal > 1) { event.preventDefault(); select(graph.nodes[selected.ordinal - 2].id, true); }
  });
  new ResizeObserver(() => { if (graph) fit(); }).observe(viewport);
  const state = new URLSearchParams(location.hash.slice(1));
  showView(state.get('view'), state.get('node'));
  window.__reader = { data, graph: () => graph, selected: () => selected, showView, select };
})();
