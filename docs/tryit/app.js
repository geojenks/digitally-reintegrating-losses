/* Try-it page: loss masking + procedural stitch preview.
   Vanilla JS, no dependencies. All work happens in the browser. */
(function () {
  'use strict';

  var STITCHES = ['french_knot', 'satin', 'silk_purl'];
  var OVERLAY = [[220, 40, 40], [40, 90, 230], [225, 175, 20]]; // red, blue, gold
  var OVERLAY_ALPHA = 0.45;
  var MAX_DIM = 1024;
  var UNDO_LIMIT = 15;

  function $(id) { return document.getElementById(id); }

  var view = $('view'), ui = $('ui'), stage = $('stage');
  var vctx = view.getContext('2d');
  var uctx = ui.getContext('2d');

  var W = 0, H = 0;
  var baseData = null;          // ImageData of working-resolution image
  var stem = 'image';
  var masks = [null, null, null]; // Uint8Array(W*H) per stitch
  var textures = [null, null, null]; // Uint8ClampedArray RGBA per stitch
  var texKeys = ['', '', ''];
  var outData = null;           // ImageData reused for compositing
  var undoStack = [];
  var cur = 0;                  // current stitch index
  var renderPending = false;

  // ---------- PRNG ----------
  function mulberry32(a) {
    a = a >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randint(rng, lo, hi) { // inclusive
    return lo + Math.floor(rng() * (hi - lo + 1));
  }
  function hexToRgb(hex) {
    var n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function clip(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // Adds gaussian noise (Box-Muller) to a float RGB buffer and packs to RGBA bytes.
  function noiseAndPack(buf, sigma, rng) {
    var n = W * H;
    var out = new Uint8ClampedArray(n * 4);
    var spare = 0, hasSpare = false;
    for (var i = 0, j = 0, k = 0; i < n; i++, j += 3, k += 4) {
      for (var c = 0; c < 3; c++) {
        var g;
        if (hasSpare) { g = spare; hasSpare = false; }
        else {
          var u1 = rng(); if (u1 < 1e-12) u1 = 1e-12;
          var u2 = rng();
          var mag = Math.sqrt(-2 * Math.log(u1));
          g = mag * Math.cos(2 * Math.PI * u2);
          spare = mag * Math.sin(2 * Math.PI * u2); hasSpare = true;
        }
        out[k + c] = clip(buf[j + c] + g * sigma, 0, 255); // Uint8ClampedArray rounds
      }
      out[k + 3] = 255;
    }
    return out;
  }

  // ---------- Texture generators (ports of the paper's Python) ----------
  function texFrenchKnot(base, r, rng) {
    var buf = new Float32Array(W * H * 3);
    var i, p;
    for (i = 0; i < W * H; i++) {
      buf[i * 3] = base[0] * 0.55; buf[i * 3 + 1] = base[1] * 0.55; buf[i * 3 + 2] = base[2] * 0.55;
    }
    var step = Math.round(r * 1.9);
    var win = 2 * r + 1;
    var bump = new Float32Array(win * win);
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        var d = Math.sqrt(dy * dy + dx * dx) / r;
        bump[(dy + r) * win + (dx + r)] = clip(1.15 - Math.pow(d, 1.5), 0, 1);
      }
    }
    var half = Math.floor(step / 2);
    var row = 0;
    for (var y = r; y < H - r; y += step, row++) {
      var x0 = r + ((row % 2) ? half : 0);
      for (var x = x0; x < W - r; x += step) {
        var cy = y + randint(rng, -3, 3);
        var cx = x + randint(rng, -3, 3);
        if (cy < r || cy >= H - r || cx < r || cx >= W - r) continue;
        var f = 0.75 + 0.5 * rng();
        var kr = clip(base[0] * f, 0, 255), kg = clip(base[1] * f, 0, 255), kb = clip(base[2] * f, 0, 255);
        for (var by = 0; by < win; by++) {
          var rowOff = ((cy - r + by) * W + (cx - r)) * 3;
          for (var bx = 0; bx < win; bx++) {
            var a = bump[by * win + bx] * 0.95;
            if (a === 0) continue;
            p = rowOff + bx * 3;
            buf[p] = buf[p] * (1 - a) + kr * a;
            buf[p + 1] = buf[p + 1] * (1 - a) + kg * a;
            buf[p + 2] = buf[p + 2] * (1 - a) + kb * a;
          }
        }
      }
    }
    return noiseAndPack(buf, 8, rng);
  }

  function texSatin(base, tp, angleDeg, rng) {
    var buf = new Float32Array(W * H * 3);
    var th = angleDeg * Math.PI / 180;
    var ct = Math.cos(th), st = Math.sin(th);
    for (var y = 0; y < H; y++) {
      for (var x = 0; x < W; x++) {
        var phase = x * ct + y * st;
        var t = 0.82 + 0.18 * Math.abs(Math.sin(Math.PI * phase / tp));
        var p = (y * W + x) * 3;
        buf[p] = base[0] * t; buf[p + 1] = base[1] * t; buf[p + 2] = base[2] * t;
      }
    }
    return noiseAndPack(buf, 8, rng);
  }

  function texSilkPurl(base, cp, rng) {
    var buf = new Float32Array(W * H * 3);
    var S = randint(rng, 40, 90);
    var shim = new Float32Array(W);
    for (var x = 0; x < W; x++) shim[x] = 0.85 + 0.15 * Math.sin(2 * Math.PI * x / S);
    for (var y = 0; y < H; y++) {
      var coil = 0.5 + 0.5 * Math.abs(Math.sin(Math.PI * y / cp));
      var band = 0.8 + 0.2 * Math.sin(2 * Math.PI * y / (4 * cp));
      var rowF = coil * band;
      for (var x2 = 0; x2 < W; x2++) {
        var f = rowF * shim[x2];
        var p = (y * W + x2) * 3;
        buf[p] = base[0] * f; buf[p + 1] = base[1] * f; buf[p + 2] = base[2] * f;
      }
    }
    return noiseAndPack(buf, 10, rng);
  }

  function seedValue() {
    var s = parseInt($('seed').value, 10);
    return isNaN(s) ? 1600 : s;
  }
  function texParams(s) {
    if (s === 0) return { r: parseInt($('fkR').value, 10), col: $('fkCol').value };
    if (s === 1) return { tp: parseFloat($('saTp').value), ang: parseFloat($('saAng').value), col: $('saCol').value };
    return { cp: parseFloat($('spCp').value), col: $('spCol').value };
  }
  function ensureTexture(s) {
    var p = texParams(s);
    var key = W + 'x' + H + '|' + seedValue() + '|' + JSON.stringify(p);
    if (textures[s] && texKeys[s] === key) return textures[s];
    var rng = mulberry32(seedValue());
    var base = hexToRgb(p.col);
    if (s === 0) textures[s] = texFrenchKnot(base, p.r, rng);
    else if (s === 1) textures[s] = texSatin(base, p.tp, p.ang, rng);
    else textures[s] = texSilkPurl(base, p.cp, rng);
    texKeys[s] = key;
    return textures[s];
  }

  // ---------- Compositing ----------
  function maskHasAny(m) {
    if (!m) return false;
    for (var i = 0; i < m.length; i++) if (m[i]) return true;
    return false;
  }

  function composite(target, usePreview) {
    var src = baseData.data, dst = target.data;
    dst.set(src);
    var n = W * H;
    for (var s = 0; s < 3; s++) {
      var m = masks[s];
      if (!maskHasAny(m)) continue;
      if (usePreview) {
        var tex = ensureTexture(s);
        for (var i = 0; i < n; i++) {
          if (m[i]) { var k = i * 4; dst[k] = tex[k]; dst[k + 1] = tex[k + 1]; dst[k + 2] = tex[k + 2]; }
        }
      } else {
        var oc = OVERLAY[s], a = OVERLAY_ALPHA, b = 1 - a;
        var r0 = oc[0] * a, g0 = oc[1] * a, b0 = oc[2] * a;
        for (var j = 0; j < n; j++) {
          if (m[j]) {
            var q = j * 4;
            dst[q] = dst[q] * b + r0; dst[q + 1] = dst[q + 1] * b + g0; dst[q + 2] = dst[q + 2] * b + b0;
          }
        }
      }
    }
    return target;
  }

  function render() {
    renderPending = false;
    if (!baseData) return;
    composite(outData, $('previewToggle').checked);
    vctx.putImageData(outData, 0, 0);
    updateMaskInfo();
  }
  function requestRender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(render);
  }

  function updateMaskInfo() {
    var parts = [];
    for (var s = 0; s < 3; s++) {
      var c = 0, m = masks[s];
      if (m) for (var i = 0; i < m.length; i++) c += m[i];
      parts.push(STITCHES[s] + ': ' + (W * H ? (100 * c / (W * H)).toFixed(1) : '0.0') + '%');
    }
    $('maskInfo').textContent = 'Coverage: ' + parts.join(', ');
  }

  // ---------- Image loading ----------
  function setStatus(t) { $('status').textContent = t || ''; }

  function loadFromImage(img, name) {
    var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    var sc = Math.min(1, MAX_DIM / Math.max(w, h));
    W = Math.max(1, Math.round(w * sc)); H = Math.max(1, Math.round(h * sc));
    view.width = W; view.height = H; ui.width = W; ui.height = H;
    vctx.imageSmoothingQuality = 'high';
    vctx.clearRect(0, 0, W, H);
    vctx.drawImage(img, 0, 0, W, H);
    try {
      baseData = vctx.getImageData(0, 0, W, H);
    } catch (e) {
      baseData = null;
      stage.classList.add('empty');
      setStatus('Could not read the image pixels (browser security). If you opened this page from disk, serve it over http instead.');
      return;
    }
    outData = vctx.createImageData(W, H);
    masks = [new Uint8Array(W * H), new Uint8Array(W * H), new Uint8Array(W * H)];
    textures = [null, null, null]; texKeys = ['', '', ''];
    undoStack = [];
    stem = (name || 'image').replace(/\.[^.]*$/, '').replace(/[^A-Za-z0-9_\-]+/g, '_') || 'image';
    stage.classList.remove('empty');
    $('imgInfo').textContent = 'Loaded "' + (name || 'image') + '" at ' + W + ' x ' + H + ' px (from ' + w + ' x ' + h + '). Export stem: ' + stem;
    setStatus('');
    requestRender();
  }

  function loadFile(file) {
    if (!file || !/^image\//.test(file.type)) { setStatus('Please choose an image file.'); return; }
    var url = URL.createObjectURL(file);
    var img = new Image();
    img.onload = function () { loadFromImage(img, file.name); URL.revokeObjectURL(url); };
    img.onerror = function () { setStatus('Could not decode that image.'); URL.revokeObjectURL(url); };
    img.src = url;
  }

  $('fileInput').addEventListener('change', function (e) {
    if (e.target.files && e.target.files[0]) loadFile(e.target.files[0]);
    e.target.value = '';
  });
  $('sampleBtn').addEventListener('click', function () {
    var img = new Image();
    img.onload = function () { loadFromImage(img, 'castle.jpg'); };
    img.onerror = function () { setStatus('Could not load the sample image (../assets/castle.jpg).'); };
    img.src = '../assets/castle.jpg';
  });
  ['dragenter', 'dragover'].forEach(function (ev) {
    stage.addEventListener(ev, function (e) { e.preventDefault(); stage.classList.add('dragover'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    stage.addEventListener(ev, function (e) { e.preventDefault(); stage.classList.remove('dragover'); });
  });
  stage.addEventListener('drop', function (e) {
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) loadFile(f);
  });

  // ---------- Undo ----------
  function pushUndo() {
    if (!baseData) return;
    undoStack.push(masks.map(function (m) { return m.slice(); }));
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  }
  function undo() {
    if (!undoStack.length) { setStatus('Nothing to undo.'); return; }
    masks = undoStack.pop();
    setStatus('');
    requestRender();
  }
  $('undoBtn').addEventListener('click', undo);

  // ---------- Stitch / tool selection ----------
  Array.prototype.forEach.call(document.querySelectorAll('input[name=stitch]'), function (el) {
    el.addEventListener('change', function () { if (el.checked) cur = parseInt(el.value, 10); });
  });
  function currentTool() { return document.querySelector('input[name=tool]:checked').value; }
  function updateToolOpts() {
    var t = currentTool();
    $('brushOpts').style.display = t === 'brush' ? '' : 'none';
    $('wandOpts').style.display = t === 'wand' ? '' : 'none';
    uctx.clearRect(0, 0, ui.width, ui.height);
  }
  Array.prototype.forEach.call(document.querySelectorAll('input[name=tool]'), function (el) {
    el.addEventListener('change', updateToolOpts);
  });
  updateToolOpts();

  $('clearBtn').addEventListener('click', function () {
    if (!baseData) return;
    pushUndo();
    masks[cur].fill(0);
    requestRender();
  });

  function bindOut(inputId, outId) {
    $(inputId).addEventListener('input', function () { $(outId).textContent = $(inputId).value; });
  }
  bindOut('brushR', 'brushOut');
  bindOut('tol', 'tolOut');
  bindOut('fkR', 'fkROut');
  bindOut('saTp', 'saTpOut');
  bindOut('saAng', 'saAngOut');
  bindOut('spCp', 'spCpOut');
  ['fkR', 'fkCol', 'saTp', 'saAng', 'saCol', 'spCp', 'spCol', 'seed', 'previewToggle'].forEach(function (id) {
    $(id).addEventListener('input', requestRender);
    $(id).addEventListener('change', requestRender);
  });

  // ---------- Mask operations ----------
  function stampCircle(m, cx, cy, rad, v) {
    var x0 = Math.max(0, Math.floor(cx - rad)), x1 = Math.min(W - 1, Math.ceil(cx + rad));
    var y0 = Math.max(0, Math.floor(cy - rad)), y1 = Math.min(H - 1, Math.ceil(cy + rad));
    var r2 = rad * rad;
    for (var y = y0; y <= y1; y++) {
      var dy = y - cy, row = y * W;
      for (var x = x0; x <= x1; x++) {
        var dx = x - cx;
        if (dx * dx + dy * dy <= r2) m[row + x] = v;
      }
    }
  }
  function strokeLine(m, ax, ay, bx, by, rad, v) {
    var dist = Math.hypot(bx - ax, by - ay);
    var step = Math.max(1, rad / 3);
    var n = Math.max(1, Math.ceil(dist / step));
    for (var i = 1; i <= n; i++) {
      var t = i / n;
      stampCircle(m, ax + (bx - ax) * t, ay + (by - ay) * t, rad, v);
    }
  }
  function fillRect(m, ax, ay, bx, by, v) {
    var x0 = clip(Math.floor(Math.min(ax, bx)), 0, W - 1), x1 = clip(Math.floor(Math.max(ax, bx)), 0, W - 1);
    var y0 = clip(Math.floor(Math.min(ay, by)), 0, H - 1), y1 = clip(Math.floor(Math.max(ay, by)), 0, H - 1);
    for (var y = y0; y <= y1; y++) m.fill(v, y * W + x0, y * W + x1 + 1);
  }

  // Colour-match map: 1 where pixel is within tolerance (RGB Euclidean) of the seed colour.
  function matchMap(sx, sy, tol) {
    var d = baseData.data, n = W * H;
    var k = (sy * W + sx) * 4;
    var r0 = d[k], g0 = d[k + 1], b0 = d[k + 2];
    var t2 = tol * tol;
    var mm = new Uint8Array(n);
    for (var i = 0, q = 0; i < n; i++, q += 4) {
      var dr = d[q] - r0, dg = d[q + 1] - g0, db = d[q + 2] - b0;
      mm[i] = (dr * dr + dg * dg + db * db <= t2) ? 1 : 0;
    }
    return mm;
  }
  // Iterative scanline flood fill over the match map (no recursion).
  function floodFill(mm, sx, sy) {
    var sel = new Uint8Array(W * H);
    // A pixel can be pushed at most once from the row above and once from the row below.
    var stack = new Int32Array(2 * W * H + 2);
    var sp = 0;
    stack[sp++] = sy * W + sx;
    while (sp > 0) {
      var i = stack[--sp];
      if (sel[i]) continue;
      var y = (i / W) | 0, row = y * W, x = i - row;
      var xl = x, xr = x;
      while (xl > 0 && !sel[row + xl - 1] && mm[row + xl - 1]) xl--;
      while (xr < W - 1 && !sel[row + xr + 1] && mm[row + xr + 1]) xr++;
      for (var xx = xl; xx <= xr; xx++) sel[row + xx] = 1;
      for (var dir = -1; dir <= 1; dir += 2) {
        var ny = y + dir;
        if (ny < 0 || ny >= H) continue;
        var nrow = ny * W, prev = false;
        for (var x2 = xl; x2 <= xr; x2++) {
          var idx = nrow + x2;
          var ok = mm[idx] && !sel[idx];
          if (ok && !prev) { if (sp < stack.length) stack[sp++] = idx; }
          prev = ok;
        }
      }
    }
    return sel;
  }
  function applySelection(m, sel, v) {
    for (var i = 0; i < m.length; i++) if (sel[i]) m[i] = v;
  }

  // ---------- Pointer handling ----------
  var drag = null; // {tool, x0, y0, lx, ly, v}
  var hover = null;

  function toImage(e) {
    var r = view.getBoundingClientRect();
    return { x: (e.clientX - r.left) * W / r.width, y: (e.clientY - r.top) * H / r.height };
  }
  function subtracting(e) { return e.altKey || $('eraser').checked; }

  function drawUI() {
    uctx.clearRect(0, 0, ui.width, ui.height);
    var scale = W / view.getBoundingClientRect().width || 1;
    uctx.lineWidth = Math.max(1, scale);
    if (drag && drag.tool === 'box') {
      uctx.setLineDash([6 * scale, 4 * scale]);
      uctx.strokeStyle = '#000';
      uctx.strokeRect(drag.x0, drag.y0, drag.lx - drag.x0, drag.ly - drag.y0);
      uctx.strokeStyle = '#fff';
      uctx.lineDashOffset = 6 * scale;
      uctx.strokeRect(drag.x0, drag.y0, drag.lx - drag.x0, drag.ly - drag.y0);
      uctx.setLineDash([]); uctx.lineDashOffset = 0;
    }
    if (hover && currentTool() === 'brush') {
      var rad = parseInt($('brushR').value, 10);
      uctx.beginPath(); uctx.arc(hover.x, hover.y, rad, 0, 2 * Math.PI);
      uctx.strokeStyle = hover.sub ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.95)';
      uctx.stroke();
      uctx.beginPath(); uctx.arc(hover.x, hover.y, rad + scale, 0, 2 * Math.PI);
      uctx.strokeStyle = hover.sub ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)';
      uctx.stroke();
    }
  }

  view.addEventListener('pointerdown', function (e) {
    if (!baseData || e.button !== 0) return;
    e.preventDefault();
    var p = toImage(e), tool = currentTool(), v = subtracting(e) ? 0 : 1;
    if (tool === 'wand') {
      var sx = clip(Math.floor(p.x), 0, W - 1), sy = clip(Math.floor(p.y), 0, H - 1);
      var t0 = performance.now();
      var mm = matchMap(sx, sy, parseFloat($('tol').value));
      var sel = $('nonContig').checked ? mm : floodFill(mm, sx, sy);
      pushUndo();
      applySelection(masks[cur], sel, v);
      setStatus('Magic wand: ' + (performance.now() - t0).toFixed(0) + ' ms');
      requestRender();
      return;
    }
    pushUndo();
    view.setPointerCapture(e.pointerId);
    drag = { tool: tool, x0: p.x, y0: p.y, lx: p.x, ly: p.y, v: v };
    if (tool === 'brush') {
      stampCircle(masks[cur], p.x, p.y, parseInt($('brushR').value, 10), v);
      requestRender();
    }
    drawUI();
  });
  view.addEventListener('pointermove', function (e) {
    if (!baseData) return;
    var p = toImage(e);
    hover = { x: p.x, y: p.y, sub: subtracting(e) };
    if (drag) {
      if (drag.tool === 'brush') {
        strokeLine(masks[cur], drag.lx, drag.ly, p.x, p.y, parseInt($('brushR').value, 10), drag.v);
        requestRender();
      }
      drag.lx = p.x; drag.ly = p.y;
    }
    drawUI();
  });
  function endDrag(e) {
    if (!drag) return;
    if (drag.tool === 'box') {
      var p = toImage(e);
      fillRect(masks[cur], drag.x0, drag.y0, p.x, p.y, drag.v);
      requestRender();
    }
    drag = null;
    drawUI();
  }
  view.addEventListener('pointerup', endDrag);
  view.addEventListener('pointercancel', function () { drag = null; drawUI(); });
  view.addEventListener('pointerleave', function () { hover = null; if (!drag) drawUI(); });

  // Alt would otherwise focus the browser menu on Windows; Ctrl+Z for undo.
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Alt') e.preventDefault();
    var tag = (e.target && e.target.tagName) || '';
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z') && tag !== 'INPUT' && tag !== 'TEXTAREA') {
      e.preventDefault();
      undo();
    }
    if (hover) { hover.sub = e.altKey || $('eraser').checked; drawUI(); }
  });
  document.addEventListener('keyup', function (e) {
    if (e.key === 'Alt') e.preventDefault();
    if (hover) { hover.sub = e.altKey || $('eraser').checked; drawUI(); }
  });

  // ---------- Export ----------
  function downloadCanvas(canvas, filename) {
    canvas.toBlob(function (blob) {
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
    }, 'image/png');
  }
  function canvasFrom(imgData) {
    var c = document.createElement('canvas');
    c.width = W; c.height = H;
    c.getContext('2d').putImageData(imgData, 0, 0);
    return c;
  }
  function needImage() {
    if (!baseData) { setStatus('Load an image first.'); return false; }
    return true;
  }

  $('dlImage').addEventListener('click', function () {
    if (!needImage()) return;
    downloadCanvas(canvasFrom(baseData), stem + '.png');
  });
  $('dlMasks').addEventListener('click', function () {
    if (!needImage()) return;
    var saved = [], delay = 0;
    for (var s = 0; s < 3; s++) {
      if (!maskHasAny(masks[s])) continue;
      var id = new ImageData(W, H), d = id.data, m = masks[s];
      for (var i = 0, k = 0; i < m.length; i++, k += 4) {
        var v = m[i] ? 255 : 0;
        d[k] = v; d[k + 1] = v; d[k + 2] = v; d[k + 3] = 255;
      }
      var name = stem + '__' + STITCHES[s] + '.png';
      saved.push(name);
      (function (c, n, t) { setTimeout(function () { downloadCanvas(c, n); }, t); })(canvasFrom(id), name, delay);
      delay += 400; // spaced out so browsers accept multiple downloads
    }
    setStatus(saved.length ? 'Saving: ' + saved.join(', ') : 'All masks are empty; nothing to save.');
  });
  $('dlPreview').addEventListener('click', function () {
    if (!needImage()) return;
    var id = composite(new ImageData(W, H), true);
    downloadCanvas(canvasFrom(id), stem + '__preview.png');
  });

  window.addEventListener('resize', drawUI);
})();
