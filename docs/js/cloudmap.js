/* "Checking fills against real stitches": the descriptor-space cloud map (paper Fig. 6). */
(function () {
  'use strict';
  var root = document.getElementById('cloudmap');
  if (!root) return;
  var BASE = 'assets/cloudmap/';

  var VARIANT = {
    flux1_fill: 'FLUX.1-Fill-dev',
    qwen_base: 'Qwen-Image (masked)',
    qwen_edit: 'Qwen-Image-Edit',
    sdxl_base: 'SDXL (masked img2img)',
    sdxl_inpaint: 'SDXL-inpainting',
    zimage_base: 'Z-Image base'
  };
  // Plot geometry in SVG user units.
  var W = 240, H = 262, X0 = 32, Y0 = 40, S = 196, U = S / 20;
  function sx(x) { return X0 + (x + 10) * U; }
  function sy(y) { return Y0 + (10 - y) * U; }
  var uid = 0;

  RL.whenNear(root, function () {
    RL.json(BASE + 'points.json').then(build).catch(function (e) {
      root.innerHTML = '<p class="widget-loading">Could not load the cloud-map data.</p>';
      console.warn('cloudmap:', e);
    });
  }, '600px 0px');

  function build(data) {
    var panels = data.panels, C = data.colours;
    var mqMobile = window.matchMedia('(max-width: 640px)');
    var state = { mode: mqMobile.matches ? 'focus' : 'grid', pi: 0, sel: null };

    // ---------- SVG panel ----------
    function drawPanel(p, pi, big) {
      var id = 'cmclip' + (uid++);
      var svg = RL.svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
        'aria-label': p.stitch_label + ', ' + p.title + ': mean distance LoRA off ' + p.strip.off +
          ', LoRA on ' + p.strip.on + ', ' + p.strip.delta.replace('Δ', 'improvement ') });
      svg.appendChild(RL.svg('defs', null, RL.svg('clipPath', { id: id },
        RL.svg('rect', { x: X0, y: Y0, width: S, height: S }))));
      // strip
      var fs = 12.5;
      svg.appendChild(RL.svg('text', { x: X0, y: 26, 'font-size': fs, 'font-family': 'ui-monospace,Menlo,Consolas,monospace' }, [
        RL.svg('tspan', { fill: C.off, text: 'off ' + p.strip.off }),
        RL.svg('tspan', { dx: 8, fill: C.on, text: 'on ' + p.strip.on }),
        RL.svg('tspan', { dx: 8, fill: '#222', 'font-weight': 600, text: p.strip.delta })
      ]));
      // frame + grid + ticks
      var axes = RL.svg('g');
      p.ticks.forEach(function (tv, k) {
        if (tv !== -10 && tv !== 10) {
          axes.appendChild(RL.svg('line', { x1: sx(tv), x2: sx(tv), y1: Y0, y2: Y0 + S, stroke: '#e6e2da', 'stroke-width': 0.6 }));
          axes.appendChild(RL.svg('line', { y1: sy(tv), y2: sy(tv), x1: X0, x2: X0 + S, stroke: '#e6e2da', 'stroke-width': 0.6 }));
        }
        var lbl = p.tick_labels[k];
        if (lbl) {
          axes.appendChild(RL.svg('text', { class: 'ticklbl', x: sx(tv), y: Y0 + S + 11, 'font-size': 8.5, fill: '#666', 'text-anchor': 'middle', text: lbl }));
          axes.appendChild(RL.svg('text', { class: 'ticklbl', x: X0 - 4, y: sy(tv) + 3, 'font-size': 8.5, fill: '#666', 'text-anchor': 'end', text: lbl }));
        }
      });
      axes.appendChild(RL.svg('text', { class: 'axlbl', x: X0 + S / 2, y: H - 3, 'font-size': 9.5, fill: '#444', 'text-anchor': 'middle', text: 'whitened ' + p.xlabel }));
      axes.appendChild(RL.svg('text', { class: 'axlbl', x: 0, y: 0, 'font-size': 9.5, fill: '#444', 'text-anchor': 'middle',
        transform: 'translate(9 ' + (Y0 + S / 2) + ') rotate(-90)', text: 'whitened ' + p.ylabel.replace(p.stitch_label + ' ', '') }));
      svg.appendChild(axes);

      var g = RL.svg('g', { 'clip-path': 'url(#' + id + ')' });
      svg.appendChild(g);
      var rs = big ? 1.25 : 1;
      g.appendChild(RL.svg('circle', { cx: sx(0), cy: sy(0), r: p.real_radius * U, fill: 'none', stroke: '#777',
        'stroke-width': 0.9, 'stroke-dasharray': '3 2.5' }));
      var realSel = {};
      p.real_sel.forEach(function (r, j) { realSel[r.i] = j; });
      p.real.forEach(function (xy, i) {
        if (i in realSel) return;
        g.appendChild(RL.svg('circle', { cx: sx(xy[0]).toFixed(2), cy: sy(xy[1]).toFixed(2), r: 1.35 * rs,
          fill: C.real, 'fill-opacity': 0.6, 'data-k': 'r' + i }));
      });
      [0, 1].forEach(function (s) {
        p.fills.forEach(function (f, i) {
          if (f.s !== s || f.img) return;
          g.appendChild(RL.svg('circle', { cx: sx(f.x).toFixed(2), cy: sy(f.y).toFixed(2), r: 1.7 * rs,
            fill: s ? C.on : C.off, 'fill-opacity': 0.78, 'data-k': 'f' + i }));
        });
      });
      // mean-distance circles and centres, as in the figure
      g.appendChild(RL.svg('circle', { cx: sx(p.com_off[0]), cy: sy(p.com_off[1]), r: p.mean_D_off * U, fill: 'none', stroke: C.off, 'stroke-width': 1.4 }));
      g.appendChild(RL.svg('circle', { cx: sx(p.com_on[0]), cy: sy(p.com_on[1]), r: p.mean_D_on * U, fill: 'none', stroke: C.on, 'stroke-width': 1.4 }));
      g.appendChild(centre(RL.svg('path', { d: starPath(sx(0), sy(0), 4.6, 2), fill: '#222' }), sx(0), sy(0), 'c2'));
      g.appendChild(centre(crossMark(sx(p.com_off[0]), sy(p.com_off[1]), 3.6, C.off), sx(p.com_off[0]), sy(p.com_off[1]), 'c0'));
      g.appendChild(centre(crossMark(sx(p.com_on[0]), sy(p.com_on[1]), 3.6, C.on), sx(p.com_on[0]), sy(p.com_on[1]), 'c1'));
      // selectable dots on top
      var selG = RL.svg('g', { class: 'sel-layer' });
      p.real_sel.forEach(function (r, j) {
        var xy = p.real[r.i];
        selG.appendChild(RL.svg('circle', { class: 'sel', cx: sx(xy[0]), cy: sy(xy[1]), r: big ? 4.2 : 3.4, fill: C.real,
          stroke: '#222', 'stroke-width': 1.1, 'data-k': 's' + j }));
      });
      p.fills.forEach(function (f, i) {
        if (!f.img) return;
        selG.appendChild(RL.svg('circle', { class: 'sel', cx: sx(f.x), cy: sy(f.y), r: big ? 4.2 : 3.4, fill: f.s ? C.on : C.off,
          stroke: '#222', 'stroke-width': 1.1, 'data-k': 'f' + i }));
      });
      g.appendChild(selG);
      g.appendChild(RL.svg('g', { class: 'hl-layer' }));
      svg.appendChild(RL.svg('rect', { x: X0, y: Y0, width: S, height: S, fill: 'none', stroke: '#bbb', 'stroke-width': 0.8 }));
      svg._pi = pi;
      return svg;
    }
    function starPath(cx, cy, R, r) {
      var d = '';
      for (var k = 0; k < 10; k++) {
        var a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r : R;
        d += (k ? 'L' : 'M') + (cx + rr * Math.cos(a)).toFixed(2) + ' ' + (cy + rr * Math.sin(a)).toFixed(2);
      }
      return d + 'Z';
    }
    // A centre marker plus a larger invisible hover/click target carrying its key.
    function centre(mark, cx, cy, key) {
      return RL.svg('g', { class: 'cm-centre', 'data-k': key }, [mark,
        RL.svg('circle', { cx: cx, cy: cy, r: 6.5, fill: 'transparent' })]);
    }
    function crossMark(cx, cy, r, col) {
      var d = 'M' + (cx - r) + ' ' + (cy - r) + 'L' + (cx + r) + ' ' + (cy + r) + 'M' + (cx - r) + ' ' + (cy + r) + 'L' + (cx + r) + ' ' + (cy - r);
      return RL.svg('g', null, [RL.svg('path', { d: d, stroke: '#222', 'stroke-width': 3.2, 'stroke-linecap': 'round' }),
        RL.svg('path', { d: d, stroke: col, 'stroke-width': 1.8, 'stroke-linecap': 'round' })]);
    }

    function dotXY(p, key) {
      var t = key[0], i = +key.slice(1);
      if (t === 'f') return [p.fills[i].x, p.fills[i].y];
      if (t === 's') return p.real[p.real_sel[i].i];
      return p.real[i];
    }
    function tipText(p, key) {
      var t = key[0], i = +key.slice(1);
      if (t === 'f') {
        var f = p.fills[i];
        return (f.s ? 'LoRA on (s = 1)' : 'LoRA off (s = 0)') + ' · ' + (VARIANT[f.variant] || f.variant) + ' · D ' + RL.fmt(f.D);
      }
      if (t === 's') return 'Real reference crop · D ' + RL.fmt(p.real_sel[i].D);
      if (key === 'c0' || key === 'c1') {
        var on = key === 'c1';
        return 'Mean position (centre) of the LoRA ' + (on ? 'on' : 'off') + ' fills\n' +
          'circle radius = their mean D, ' + RL.fmt(on ? p.mean_D_on : p.mean_D_off);
      }
      if (key === 'c2') return 'Centre of the real stitches (D = 0)';
      return 'Real reference crop';
    }
    // Keys to highlight for a selection: a fill and its paired partner, or one real dot.
    function selKeys(p, key) {
      if (key[0] === 'f') { var f = p.fills[+key.slice(1)]; return [key, 'f' + f.pair]; }
      return [key];
    }

    // ---------- layout ----------
    var toolbar = RL.el('div', { class: 'cm-toolbar' });
    var backBtn = RL.el('button', { type: 'button', class: 'iconbtn cm-mobile-hide', text: '← All nine panels' });
    var hint = RL.el('span', { class: 'muted', style: 'font-size:.85em', text: 'Click a panel to enlarge it and pick dots.' });
    var stitchSeg = RL.el('div', { class: 'seg', role: 'group', 'aria-label': 'Stitch type' });
    var sizeSeg = RL.el('div', { class: 'seg', role: 'group', 'aria-label': 'Loss size' });
    var stitchBtns = data.rows.map(function (st, r) {
      var b = RL.el('button', { type: 'button', text: panels[r * 3].stitch_label });
      b.addEventListener('click', function () { interact(); focusRandom(r * 3 + (state.pi % 3)); });
      stitchSeg.appendChild(b); return b;
    });
    var sizeBtns = data.cols.map(function (sz, c) {
      var b = RL.el('button', { type: 'button', text: panels[c].hole_px + ' px' });
      b.addEventListener('click', function () { interact(); focusRandom(Math.floor(state.pi / 3) * 3 + c); });
      sizeSeg.appendChild(b); return b;
    });
    toolbar.appendChild(backBtn);
    toolbar.appendChild(hint);
    toolbar.appendChild(stitchSeg);
    toolbar.appendChild(sizeSeg);

    var legend = RL.el('ul', { class: 'cm-legend' }, [
      legendItem('<circle cx="9" cy="7" r="3" fill="' + C.real + '"/>', 'real stitches'),
      legendItem('<circle cx="9" cy="7" r="3" fill="' + C.off + '"/>', 'LoRA off (s = 0)'),
      legendItem('<circle cx="9" cy="7" r="3" fill="' + C.on + '"/>', 'LoRA on (s = 1)'),
      legendItem('<circle cx="9" cy="7" r="6" fill="none" stroke="#777" stroke-dasharray="3 2.5"/>', 'real self-distance'),
      legendItem('<path d="M1 3.5L7 10.5M1 10.5L7 3.5" stroke="' + C.off + '" stroke-width="1.8" stroke-linecap="round"/>' +
        '<path d="M11 3.5L17 10.5M11 10.5L17 3.5" stroke="' + C.on + '" stroke-width="1.8" stroke-linecap="round"/>', 'mean position (off / on)'),
      legendItem('<circle cx="5" cy="7" r="4.2" fill="none" stroke="' + C.off + '" stroke-width="1.4"/>' +
        '<circle cx="13" cy="7" r="4.2" fill="none" stroke="' + C.on + '" stroke-width="1.4"/>', 'mean D (off / on)'),
      legendItem('<circle cx="9" cy="7" r="4" fill="' + C.on + '" stroke="#222" stroke-width="1.2"/>', 'selectable example')
    ]);
    function legendItem(svgInner, label) {
      var li = RL.el('li');
      li.innerHTML = '<svg viewBox="0 0 18 14" aria-hidden="true">' + svgInner + '</svg>';
      li.appendChild(document.createTextNode(label));
      return li;
    }

    var view = RL.el('div', { class: 'cm-view' });
    var tip = RL.el('div', { class: 'cm-tip', 'aria-hidden': 'true' });
    var gridEl = RL.el('div', { class: 'cm-grid cm-small' });
    var focusWrap = RL.el('div');
    var card = RL.el('aside', { class: 'cm-card', 'aria-live': 'polite' });
    var gridSvgs = [];

    gridEl.appendChild(RL.el('div'));
    data.cols.forEach(function (sz, c) { gridEl.appendChild(RL.el('div', { class: 'colhead', text: panels[c].title })); });
    data.rows.forEach(function (st, r) {
      gridEl.appendChild(RL.el('div', { class: 'rowhead', text: panels[r * 3].stitch_label }));
      for (var c = 0; c < 3; c++) {
        (function (pi) {
          var p = panels[pi], svg = drawPanel(p, pi, false);
          var btn = RL.el('button', { type: 'button', class: 'cm-panel',
            'aria-label': 'Enlarge ' + p.stitch_label + ', ' + p.title + ' (off ' + p.strip.off + ', on ' + p.strip.on + ')' }, svg);
          btn.addEventListener('click', function (e) {
            interact();
            var k = e.target.getAttribute && e.target.getAttribute('data-k');
            if (k && e.target.classList.contains('sel')) { focus(pi); select(pi, k); }
            else focusRandom(pi);
          });
          gridSvgs[pi] = svg;
          gridEl.appendChild(btn);
        })(r * 3 + c);
      }
    });

    view.appendChild(gridEl);
    view.appendChild(focusWrap);
    view.appendChild(tip);
    root.innerHTML = '';
    root.appendChild(toolbar);
    root.appendChild(legend);
    root.appendChild(RL.el('div', { class: 'cm-body' }, [view, card]));

    // ---------- tooltips (delegated) ----------
    var tipTarget = null;
    function keyed(el) { return el && el.closest ? el.closest('[data-k]') : null; }
    function showTip(el) {
      var svg = el && el.ownerSVGElement;
      if (!svg) return;
      tipTarget = el;
      tip.textContent = tipText(panels[svg._pi], el.getAttribute('data-k'));
      var vr = view.getBoundingClientRect(), br = el.getBoundingClientRect();
      var x = br.left + br.width / 2 - vr.left, half = tip.offsetWidth / 2 || 70;
      tip.style.left = Math.max(half, Math.min(vr.width - half, x)) + 'px';
      tip.style.top = (br.top - vr.top) + 'px';
      tip.style.display = 'block';
      half = tip.offsetWidth / 2;
      tip.style.left = Math.max(half, Math.min(vr.width - half, x)) + 'px';
    }
    view.addEventListener('mouseover', function (e) { showTip(keyed(e.target)); });
    view.addEventListener('mouseout', function (e) {
      if (tipTarget && !tipTarget.contains(e.relatedTarget)) { tip.style.display = 'none'; tipTarget = null; }
    });

    // ---------- focus view ----------
    var focusSvg = null, picks = null;
    function renderFocus() {
      focusWrap.innerHTML = '';
      var p = panels[state.pi];
      focusSvg = drawPanel(p, state.pi, true);
      var box = RL.el('div', { class: 'cm-focus' }, focusSvg);
      focusSvg.addEventListener('click', function (e) {
        var k = e.target.getAttribute && e.target.getAttribute('data-k');
        if (k && e.target.classList.contains('sel')) { interact(); select(state.pi, k); return; }
        var c = keyed(e.target);   // a tap on a centre marker shows its label (touch screens)
        if (c && c.classList.contains('cm-centre')) { interact(); showTip(c); }
      });
      picks = RL.el('div', { class: 'cm-picks', role: 'group', 'aria-label': 'Selectable examples in this panel' },
        RL.el('span', { class: 'lbl', text: p.stitch_label + ', ' + p.title + '. Selectable examples:' }));
      p.fills.forEach(function (f, i) {
        if (!f.img || f.s !== 0) return;
        var b = RL.el('button', { type: 'button', 'data-k': 'f' + i, text: VARIANT[f.variant] || f.variant,
          'aria-label': (VARIANT[f.variant] || f.variant) + ': LoRA off and on pair, tile ' + f.id });
        b.addEventListener('click', function () { interact(); select(state.pi, 'f' + i); });
        picks.appendChild(b);
      });
      p.real_sel.forEach(function (r, j) {
        var b = RL.el('button', { type: 'button', 'data-k': 's' + j, text: 'Real crop' });
        b.addEventListener('click', function () { interact(); select(state.pi, 's' + j); });
        picks.appendChild(b);
      });
      focusWrap.appendChild(box);
      focusWrap.appendChild(picks);
    }

    function applyMode() {
      var focusMode = state.mode === 'focus';
      gridEl.hidden = focusMode;
      focusWrap.hidden = !focusMode;
      backBtn.hidden = !focusMode;
      hint.hidden = focusMode;
      stitchSeg.hidden = !focusMode;
      sizeSeg.hidden = !focusMode;
      var r = Math.floor(state.pi / 3), c = state.pi % 3;
      stitchBtns.forEach(function (b, j) { b.setAttribute('aria-pressed', j === r ? 'true' : 'false'); });
      sizeBtns.forEach(function (b, j) { b.setAttribute('aria-pressed', j === c ? 'true' : 'false'); });
      tip.style.display = 'none';
    }
    function focus(pi) {
      var changed = state.mode !== 'focus' || state.pi !== pi;
      state.mode = 'focus'; state.pi = pi;
      if (changed || !focusSvg) renderFocus();
      applyMode();
      if (state.sel && state.sel.pi !== pi) { state.sel = null; renderCard(); }
      highlight();
    }
    // Switching panel with the stitch / size buttons also shows a random selectable dot from it.
    function focusRandom(pi) {
      var p = panels[pi], keys = [];
      p.fills.forEach(function (f, i) { if (f.img) keys.push('f' + i); });
      p.real_sel.forEach(function (r, j) { keys.push('s' + j); });
      focus(pi);
      if (keys.length) select(pi, keys[Math.floor(Math.random() * keys.length)]);
    }
    function toGrid() {
      if (mqMobile.matches) return;
      state.mode = 'grid';
      applyMode();
      highlight();
    }
    backBtn.addEventListener('click', function () { interact(); toGrid(); });
    var onMq = function () { if (mqMobile.matches && state.mode === 'grid') focus(state.pi); };
    if (mqMobile.addEventListener) mqMobile.addEventListener('change', onMq); else if (mqMobile.addListener) mqMobile.addListener(onMq);

    // ---------- selection + highlight ----------
    function select(pi, key) {
      state.sel = { pi: pi, key: key };
      if (state.mode === 'focus' && state.pi !== pi) focus(pi);
      state.pi = pi;
      highlight();
      renderCard();
    }
    // Line from a LoRA-off fill to its LoRA-on partner, arrowhead at the on end.
    // The end labels go only in the enlarged view, where they are legible.
    function pairArrow(layer, p, key, big) {
      var f = p.fills[+key.slice(1)], g = p.fills[f.pair], off = f.s === 0 ? f : g, on = f.s === 0 ? g : f;
      var ax = sx(off.x), ay = sy(off.y), bx = sx(on.x), by = sy(on.y);
      var dx = bx - ax, dy = by - ay, L = Math.sqrt(dx * dx + dy * dy);
      if (L < 1e-6) return;
      var ux = dx / L, uy = dy / L, r0 = 9, r1 = 8;
      if (L > r0 + r1 + 4) {
        var x0 = ax + ux * r0, y0 = ay + uy * r0, x1 = bx - ux * r1, y1 = by - uy * r1, h = big ? 5 : 4, w = h * 0.55;
        layer.appendChild(RL.svg('line', { x1: x0, y1: y0, x2: x1 - ux * h * 0.6, y2: y1 - uy * h * 0.6, stroke: '#222', 'stroke-width': 1.2 }));
        layer.appendChild(RL.svg('path', { fill: '#222', d: 'M' + x1 + ' ' + y1 +
          'L' + (x1 - ux * h - uy * w) + ' ' + (y1 - uy * h + ux * w) +
          'L' + (x1 - ux * h + uy * w) + ' ' + (y1 - uy * h - ux * w) + 'Z' }));
      }
      if (!big) return;
      // Each label sits beyond its dot, pointing away from the partner, kept inside the plot.
      function label(x, y, vx, vy, text, col) {
        var tx = x + vx * 12, ty = y + vy * 12 + 3;
        var anchor = vx > 0.4 ? 'start' : vx < -0.4 ? 'end' : 'middle', wd = 34;
        var left = anchor === 'start' ? tx : anchor === 'end' ? tx - wd : tx - wd / 2;
        if (left < X0 + 2 || left + wd > X0 + S - 2) {
          // no room beside the dot: put the label under it (or over it, near the bottom edge)
          anchor = 'middle'; tx = Math.max(X0 + 2 + wd / 2, Math.min(X0 + S - 2 - wd / 2, x));
          ty = y + (y + 17 > Y0 + S - 3 || vy < -0.4 ? -11 : 17);
        }
        ty = Math.max(Y0 + 9, Math.min(Y0 + S - 3, ty));
        layer.appendChild(RL.svg('text', { x: tx, y: ty, 'font-size': 8, 'font-weight': 600, fill: col, 'text-anchor': anchor,
          stroke: '#fff', 'stroke-width': 2.6, 'paint-order': 'stroke', 'stroke-linejoin': 'round', text: text }));
      }
      label(ax, ay, -ux, -uy, 'LoRA off', C.off);
      label(bx, by, ux, uy, 'LoRA on', C.on);
    }
    function highlight() {
      var sel = state.sel;
      function paint(svg) {
        if (!svg) return;
        var layer = svg.querySelector('.hl-layer');
        layer.innerHTML = '';
        if (!sel || svg._pi !== sel.pi) return;
        var p = panels[sel.pi];
        selKeys(p, sel.key).forEach(function (k, n) {
          var xy = dotXY(p, k);
          layer.appendChild(RL.svg('circle', { class: 'hl-ring', cx: sx(xy[0]), cy: sy(xy[1]), r: n === 0 ? 8 : 6.5,
            fill: 'none', stroke: '#222', 'stroke-width': n === 0 ? 1.8 : 1.1, 'stroke-dasharray': n === 0 ? null : '2 1.6' }));
        });
        if (sel.key[0] === 'f') pairArrow(layer, p, sel.key, svg === focusSvg);
        // keep the highlighted dots above their neighbours
        selKeys(p, sel.key).forEach(function (k) {
          var d = svg.querySelector('.sel[data-k="' + k + '"]');
          if (d) d.parentNode.appendChild(d);
        });
      }
      gridSvgs.forEach(paint);
      if (focusSvg) paint(focusSvg);
      if (picks) {
        var active = sel && sel.pi === state.pi ? selKeys(panels[sel.pi], sel.key) : [];
        Array.prototype.forEach.call(picks.querySelectorAll('button'), function (b) {
          b.setAttribute('aria-pressed', active.indexOf(b.getAttribute('data-k')) >= 0 ? 'true' : 'false');
        });
      }
    }

    // The whole tile in colour with the scored region outlined, and below it the
    // normals of that region alone, enlarged; guide lines run from the box to the zoom.
    function zoomStack(img, nrm, b, altC, altN) {
      var zoom = Math.round(1 / (b[2] - b[0]));
      var crop = RL.el('div', { class: 'crop' }, [RL.el('img', { src: BASE + img, alt: altC, width: 384, height: 384 }),
        RL.el('span', { class: 'box', style: 'left:' + (b[0] * 100) + '%;top:' + (b[1] * 100) + '%;width:' +
          ((b[2] - b[0]) * 100) + '%;height:' + ((b[3] - b[1]) * 100) + '%' })]);
      var zoomed = RL.el('div', { class: 'crop nrm' }, [RL.el('img', { src: BASE + nrm, alt: altN, width: 256, height: 256 }),
        RL.el('span', { class: 'zoom', text: '×' + zoom })]);
      var guide = RL.svg('svg', { class: 'guide', viewBox: '0 0 100 210', preserveAspectRatio: 'none', 'aria-hidden': 'true' }, [
        RL.svg('path', { d: 'M' + b[0] * 100 + ' ' + b[3] * 100 + 'L0 110M' + b[2] * 100 + ' ' + b[3] * 100 + 'L100 110' })]);
      return RL.el('div', { class: 'zoomstack' }, [crop, zoomed, guide]);
    }
    function fillFig(p, f, isSel) {
      var on = f.s === 1, what = on ? 'LoRA on (s = 1)' : 'LoRA off (s = 0)';
      return RL.el('figure', { class: 'cm-fig ' + (on ? 'on' : 'off') + (isSel ? ' is-sel' : '') }, [
        zoomStack(f.img, f.nrm, f.bbox, 'The ' + what + ' fill on its whole tile, fill region outlined',
          'Normals of the ' + what + ' fill region, enlarged'),
        RL.el('figcaption', null, [
          RL.el('span', { class: 'k', style: 'background:' + (on ? C.on : C.off) }), what, RL.el('br'),
          RL.el('span', { class: 'd', text: 'D = ' + RL.fmt(f.D) })
        ])
      ]);
    }
    var autoNote = RL.el('span', { class: 'auto', text: 'Cycling through examples. Click an outlined dot to choose one.' });
    var layoutNote = 'Top: the whole tile, with the scored region outlined. Below: the normals our fine-tuned Marigold model estimates for that region, enlarged.';
    function renderCard() {
      card.innerHTML = '';
      var sel = state.sel;
      if (!sel) {
        card.appendChild(RL.el('p', { class: 'empty', text: 'Select an outlined dot, or an example button, to see the image it came from, its estimated normals and its distance D.' }));
        return;
      }
      if (!cycle.stopped) card.appendChild(autoNote);
      var p = panels[sel.pi], t = sel.key[0], i = +sel.key.slice(1);
      if (t === 'f') {
        var f = p.fills[i], g = p.fills[f.pair], off = f.s === 0 ? f : g, on = f.s === 0 ? g : f;
        card.appendChild(RL.el('h4', { text: VARIANT[f.variant] || f.variant }));
        card.appendChild(RL.el('p', { class: 'meta', text: p.stitch_label + ' · ' + p.hole_px + ' px loss · held-out tile ' + f.id }));
        card.appendChild(RL.el('div', { class: 'cm-pair' }, [fillFig(p, off, f === off), fillFig(p, on, f === on)]));
        card.appendChild(RL.el('p', { class: 'note', text: layoutNote + ' The same loss with the LoRA off and on: model, seed and mask are the same, only the strength differs. ' +
          'D goes from ' + RL.fmt(off.D) + ' to ' + RL.fmt(on.D) + '. Real crops self-score about 2.4.' }));
      } else {
        var r = p.real_sel[i];
        card.appendChild(RL.el('h4', { text: 'Real reference crop' }));
        card.appendChild(RL.el('p', { class: 'meta', text: p.stitch_label + ' · ' + r.window_px[2] + ' px window · held-out tile ' + r.id }));
        card.appendChild(RL.el('div', { class: 'cm-pair' }, [
          RL.el('figure', { class: 'cm-fig real is-sel' }, [
            zoomStack(r.img, r.nrm, r.bbox, 'Held-out tile of real ' + p.stitch_label + ' stitching, scoring window outlined',
              'Normals of the real scoring window, enlarged'),
            RL.el('figcaption', null, [RL.el('span', { class: 'k', style: 'background:' + C.real }), 'Real stitching', RL.el('br'),
              RL.el('span', { class: 'd', text: 'D = ' + RL.fmt(r.D) })])
          ]),
          RL.el('p', { class: 'note', text: 'A crop from a held-out tile, scored in the same way as the fills. Real crops self-score about 2.4, the radius of the dashed ring. ' + layoutNote })
        ]));
      }
    }

    // ---------- auto-cycle through off/on pairs until the user interacts ----------
    // Largest losses first, and within a panel the typical pair (median gain)
    // first, working outwards, so the counterexamples come last, not first.
    var order = [], pis = [0, 1, 2, 3, 4, 5, 6, 7, 8].sort(function (a, b) { return panels[b].hole_px - panels[a].hole_px || a - b; });
    var offsBy = pis.map(function (pi) {
      var fs = panels[pi].fills, offs = [];
      fs.forEach(function (f, i) { if (f.img && f.s === 0) offs.push(i); });
      offs.sort(function (a, b) { return (fs[a].D - fs[fs[a].pair].D) - (fs[b].D - fs[fs[b].pair].D); });
      var m = offs.length >> 1, out = [];
      for (var k = 0; k < offs.length; k++) out.push(offs[m + (k % 2 ? -((k + 1) >> 1) : (k >> 1))]);
      return out;
    });
    for (var round = 0; round < 5; round++) {
      pis.forEach(function (pi, j) {
        if (offsBy[j][round] !== undefined) order.push({ pi: pi, key: 'f' + offsBy[j][round] });
      });
    }
    var oi = 0;
    var cycle = new RL.AutoCycle(root, 4000, function () { advance(); });
    function advance() {
      oi = (oi + 1) % order.length;
      showAuto(order[oi]);
    }
    function showAuto(o) {
      if (state.mode === 'focus') focus(o.pi);
      select(o.pi, o.key);
      // preload the next example's crops
      var n = order[(oi + 1) % order.length], f = panels[n.pi].fills[+n.key.slice(1)], g = panels[n.pi].fills[f.pair];
      RL.loadAll([f.img, f.nrm, g.img, g.nrm].map(function (u) { return BASE + u; }));
    }
    function interact() {
      if (!cycle.stopped) { cycle.stop(); var a = card.querySelector('.auto'); if (a) a.remove(); }
    }
    root.addEventListener('mouseenter', function () { cycle.pause(); });
    root.addEventListener('mouseleave', function () { cycle.resume(); });
    root.addEventListener('keydown', interact);

    if (state.mode === 'focus') renderFocus();
    applyMode();
    if (order.length && !RL.reducedMotion()) showAuto(order[0]); else renderCard();
  }
})();
