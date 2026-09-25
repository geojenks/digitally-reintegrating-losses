/* Try-it page: layered loss masking, colour quantising, procedural stitch
   preview and job-bundle export (see pipeline/JOB_FORMAT.md).
   Vanilla JS, no dependencies. All work happens in the browser. */
(function () {
  'use strict';

  var BUILTIN = ['french_knot', 'satin', 'silk_purl'];
  var LABELS = { french_knot: 'French knot', satin: 'Satin', silk_purl: 'Silk purl', none: 'None' };
  var INIT_LABELS = { satin: 'satin', french_knot: 'French knot', silk_purl: 'silk purl', flat: 'flat' };
  var DEFAULT_COL = { french_knot: '#c89696', satin: '#eee8d6', silk_purl: '#966e3c', flat: '#b4a58c' };
  var DEFAULT_PARAMS = { thread: 4.5, angle: 0, knot_r: 13, coil: 7 };
  // Overlay tints for mask view; the first three match the old per-stitch colours.
  var TINTS = [[220, 40, 40], [40, 90, 230], [225, 175, 20], [30, 160, 70], [200, 40, 190], [20, 180, 200],
               [240, 120, 20], [120, 60, 200], [150, 200, 30], [240, 110, 160], [0, 120, 120], [140, 90, 40]];
  var OVERLAY_ALPHA = 0.45;
  var MAX_DIM = 1024;
  var MAX_LAYERS = 250;           // owner map is a Uint8Array
  var UNDO_LIMIT = 15;
  var COLAB_URL = 'https://colab.research.google.com/github/geojenks/digitally-reintegrating-losses/blob/main/notebooks/generate_textures_colab.ipynb';
  // Recipe defaults from the JOB_FORMAT.md example; seed and region_variants come from the page.
  var RECIPE = { per_region: true, proc_base: true, brim: 10, sib_feather: 3, region_max_up: 2, denoise: 0.6,
                 denoise_french_knot: 0.65, denoise_satin: 0.5, denoise_silk_purl: 0.8, size: 1024 };
  var LAYER_KEYS = ['id', 'stitch', 'mask', 'colour', 'angle', 'thread', 'knot_r', 'coil'];
  var STITCH_KEYS = ['lora', 'trigger', 'init'];

  function $(id) { return document.getElementById(id); }

  var view = $('view'), ui = $('ui'), stage = $('stage');
  var vctx = view.getContext('2d');
  var uctx = ui.getContext('2d');

  var W = 0, H = 0;
  var baseData = null;          // ImageData of working-resolution image
  var stem = 'image';
  var layers = [];              // paint order; later wins on overlap
  var activeUid = 0;
  var nextUid = 1;
  var customStitches = [];      // {name, trigger, init, lora, extra}
  var importedSettings = null;  // settings of an opened bundle, kept on re-export
  var importedOrder = null;
  var outData = null;           // ImageData reused for compositing
  var ownerBuf = null;          // Uint8Array(W*H): 1 + index of the layer that owns each pixel
  var undoStack = [];
  var renderPending = false;
  var quant = null;             // pending quantise preview
  var lastQGroup = 0, qCounter = 0;

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
  function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(function (v) { var s = clip(Math.round(v), 0, 255).toString(16); return s.length < 2 ? '0' + s : s; }).join('');
  }
  function clip(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function isHex(s) { return typeof s === 'string' && /^#[0-9a-fA-F]{6}$/.test(s); }
  function slug(s) { return String(s || '').toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, ''); }

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

  // Flat colour plus grit, for custom stitches with the "flat" init.
  function texFlat(base, rng) {
    var buf = new Float32Array(W * H * 3);
    for (var i = 0; i < W * H; i++) { buf[i * 3] = base[0]; buf[i * 3 + 1] = base[1]; buf[i * 3 + 2] = base[2]; }
    return noiseAndPack(buf, 8, rng);
  }

  function seedValue() {
    var s = parseInt($('seed').value, 10);
    return isNaN(s) ? 1600 : s;
  }

  // ---------- Stitches ----------
  function findCustom(name) {
    for (var i = 0; i < customStitches.length; i++) if (customStitches[i].name === name) return customStitches[i];
    return null;
  }
  // Procedural init style of a stitch, or null for "none" / unknown.
  function initStyle(stitch) {
    if (BUILTIN.indexOf(stitch) >= 0) return stitch;
    var c = findCustom(stitch);
    return c ? c.init : null;
  }
  function isExported(L) { return !!initStyle(L.stitch); }
  function stitchLabel(stitch) {
    if (LABELS[stitch]) return LABELS[stitch];
    return stitch + ' (custom)';
  }
  function stitchOptionsHTML(selected, withNone) {
    var all = (withNone ? ['none'] : []).concat(BUILTIN, customStitches.map(function (c) { return c.name; }));
    return all.map(function (s) {
      return '<option value="' + s + '"' + (s === selected ? ' selected' : '') + '>' + stitchLabel(s) + '</option>';
    }).join('');
  }

  // ---------- Layers ----------
  function findLayer(uid) {
    for (var i = 0; i < layers.length; i++) if (layers[i].uid === uid) return layers[i];
    return null;
  }
  function activeLayer() { return findLayer(activeUid); }
  function uniqueId(base, except) {
    base = slug(base) || 'layer';
    var taken = {};
    layers.forEach(function (L) { if (L !== except) taken[L.id] = true; });
    if (!taken[base]) return base;
    for (var k = 2; ; k++) if (!taken[base + '_' + k]) return base + '_' + k;
  }
  function nextId(stitch) {
    var taken = {};
    layers.forEach(function (L) { taken[L.id] = true; });
    for (var k = 1; ; k++) if (!taken[stitch + '_' + k]) return stitch + '_' + k;
  }
  function freeTint() {
    var used = layers.map(function (L) { return L.tint; });
    for (var i = 0; i < TINTS.length; i++) if (used.indexOf(i) < 0) return i;
    return layers.length % TINTS.length;
  }
  function defaultColour(stitch) {
    var st = initStyle(stitch);
    return DEFAULT_COL[st] || DEFAULT_COL.flat;
  }
  function makeLayer(stitch, opts) {
    opts = opts || {};
    return {
      uid: nextUid++,
      id: opts.id || nextId(stitch === 'none' ? 'layer' : stitch),
      stitch: stitch,
      colour: opts.colour || defaultColour(stitch),
      angle: DEFAULT_PARAMS.angle, thread: DEFAULT_PARAMS.thread,
      knot_r: DEFAULT_PARAMS.knot_r, coil: DEFAULT_PARAMS.coil,
      visible: true,
      tint: freeTint(),
      mask: opts.mask || new Uint8Array(W * H),
      extra: opts.extra || null,   // unknown job.json fields, kept for round trips
      qgroup: opts.qgroup || 0,
      tex: null, texKey: ''
    };
  }

  function ensureTexture(L) {
    var st = initStyle(L.stitch);
    if (!st) return null;
    var p = { st: st, col: L.colour };
    if (st === 'satin') { p.tp = L.thread; p.ang = L.angle; }
    else if (st === 'french_knot') p.r = L.knot_r;
    else if (st === 'silk_purl') p.cp = L.coil;
    var key = W + 'x' + H + '|' + seedValue() + '|' + JSON.stringify(p);
    if (L.tex && L.texKey === key) return L.tex;
    var rng = mulberry32(seedValue());
    var base = hexToRgb(L.colour);
    if (st === 'french_knot') L.tex = texFrenchKnot(base, Math.round(L.knot_r), rng);
    else if (st === 'satin') L.tex = texSatin(base, L.thread, L.angle, rng);
    else if (st === 'silk_purl') L.tex = texSilkPurl(base, L.coil, rng);
    else L.tex = texFlat(base, rng);
    L.texKey = key;
    return L.tex;
  }

  // ---------- Compositing ----------
  function maskHasAny(m) {
    if (!m) return false;
    for (var i = 0; i < m.length; i++) if (m[i]) return true;
    return false;
  }

  // Which layer owns each pixel: the last layer (in paint order) passing `want` whose mask covers it.
  function computeOwner(want, out) {
    out = out || new Uint8Array(W * H);
    out.fill(0);
    for (var li = 0; li < layers.length && li < MAX_LAYERS; li++) {
      var L = layers[li];
      if (!want(L)) continue;
      var m = L.mask, v = li + 1;
      for (var i = 0; i < m.length; i++) if (m[i]) out[i] = v;
    }
    return out;
  }
  function ensureOwnerBuf() {
    if (!ownerBuf || ownerBuf.length !== W * H) ownerBuf = new Uint8Array(W * H);
    return ownerBuf;
  }

  function composite(target, usePreview) {
    var src = baseData.data, dst = target.data;
    dst.set(src);
    var n = W * H, i, k, o;
    if (quant) { // quantise preview: flat cluster colours
      var lab = quant.labels, rgb = quant.rgb;
      for (i = 0; i < n; i++) {
        o = lab[i];
        if (o === 255) continue;
        k = i * 4; dst[k] = rgb[o * 3]; dst[k + 1] = rgb[o * 3 + 1]; dst[k + 2] = rgb[o * 3 + 2];
      }
      return target;
    }
    var own = computeOwner(usePreview ? function (L) { return L.visible && isExported(L); }
                                      : function (L) { return L.visible; }, ensureOwnerBuf());
    var texs = [], pre = [], a = OVERLAY_ALPHA, b = 1 - a;
    for (var li = 0; li < layers.length; li++) {
      var L = layers[li];
      if (usePreview) texs.push(L.visible && isExported(L) && maskHasAny(L.mask) ? ensureTexture(L) : null);
      else { var t = TINTS[L.tint % TINTS.length]; pre.push([t[0] * a, t[1] * a, t[2] * a]); }
    }
    for (i = 0; i < n; i++) {
      o = own[i];
      if (!o) continue;
      k = i * 4;
      if (usePreview) {
        var tex = texs[o - 1];
        if (tex) { dst[k] = tex[k]; dst[k + 1] = tex[k + 1]; dst[k + 2] = tex[k + 2]; }
      } else {
        var c = pre[o - 1];
        dst[k] = dst[k] * b + c[0]; dst[k + 1] = dst[k + 1] * b + c[1]; dst[k + 2] = dst[k + 2] * b + c[2];
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
    var L = activeLayer();
    if (!L || !W) { $('maskInfo').textContent = ''; return; }
    var c = 0, m = L.mask;
    for (var i = 0; i < m.length; i++) c += m[i];
    $('maskInfo').textContent = 'Coverage of "' + L.id + '": ' + (100 * c / (W * H)).toFixed(1) + '% of the image.';
  }

  // ---------- Layer list UI ----------
  function renderLayers() {
    var list = $('layerList');
    list.innerHTML = '';
    for (var li = layers.length - 1; li >= 0; li--) { // top of the list paints last
      var L = layers[li];
      var row = document.createElement('li');
      row.className = 'layer-row' + (L.uid === activeUid ? ' active' : '') +
        (isExported(L) ? '' : ' unassigned') + (L.visible ? '' : ' hidden-layer');
      row.dataset.uid = L.uid;
      var t = TINTS[L.tint % TINTS.length];
      row.innerHTML =
        '<span class="sw" title="Mask view tint" style="background:rgb(' + t.join(',') + ')"></span>' +
        '<span class="cdot" title="Init colour" style="background:' + L.colour + '"></span>' +
        '<span class="lname"></span>' +
        '<select class="lstitch" aria-label="Stitch for this layer">' + stitchOptionsHTML(L.stitch, true) + '</select>' +
        '<label class="eye" title="Show or hide"><input type="checkbox"' + (L.visible ? ' checked' : '') + ' aria-label="Show layer"></label>';
      row.querySelector('.lname').textContent = L.id;
      list.appendChild(row);
    }
    if (!layers.length) {
      var empty = document.createElement('li');
      empty.className = 'layer-empty';
      empty.textContent = W ? 'No layers yet. Add one above.' : 'Load an image first.';
      list.appendChild(empty);
    }
    renderLayerProps();
    renderClipOptions();
  }

  function renderLayerProps() {
    var L = activeLayer();
    $('layerProps').style.display = L ? '' : 'none';
    if (!L) return;
    $('lpName').value = L.id;
    $('lpCol').value = L.colour;
    var st = initStyle(L.stitch);
    $('lpSatin').style.display = st === 'satin' ? '' : 'none';
    $('lpKnot').style.display = st === 'french_knot' ? '' : 'none';
    $('lpPurl').style.display = st === 'silk_purl' ? '' : 'none';
    $('lpThread').value = L.thread; $('lpThreadOut').textContent = L.thread;
    $('lpAngle').value = L.angle; $('lpAngleOut').textContent = L.angle;
    $('lpKnotR').value = L.knot_r; $('lpKnotROut').textContent = L.knot_r;
    $('lpCoil').value = L.coil; $('lpCoilOut').textContent = L.coil;
    $('lpNote').textContent = st ? (BUILTIN.indexOf(L.stitch) < 0 ? 'Custom stitch, ' + INIT_LABELS[st] + ' init.' : '')
                                 : 'Not exported until you pick a stitch.';
    updateMaskInfo();
  }

  function renderClipOptions() {
    var sel = $('qClip'), prev = sel.value;
    var html = '<option value="">Whole image</option>';
    for (var li = layers.length - 1; li >= 0; li--) html += '<option value="' + layers[li].uid + '">' + layers[li].id + '</option>';
    sel.innerHTML = html;
    if (prev && findLayer(parseInt(prev, 10))) sel.value = prev;
  }

  function setActive(uid) {
    activeUid = uid;
    renderLayers();
  }

  $('layerList').addEventListener('click', function (e) {
    var row = e.target.closest('.layer-row');
    if (!row) return;
    var uid = parseInt(row.dataset.uid, 10);
    if (uid === activeUid) return;
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') {
      // Do not rebuild the list under an open dropdown: just move the highlight.
      activeUid = uid;
      Array.prototype.forEach.call($('layerList').children, function (r) { r.classList.toggle('active', r === row); });
      renderLayerProps();
    } else setActive(uid);
  });
  $('layerList').addEventListener('change', function (e) {
    var row = e.target.closest('.layer-row');
    if (!row) return;
    var L = findLayer(parseInt(row.dataset.uid, 10));
    if (!L) return;
    if (e.target.classList.contains('lstitch')) {
      L.stitch = e.target.value; // the layer keeps its colour (e.g. from quantising)
      activeUid = L.uid;
    } else if (e.target.type === 'checkbox') {
      L.visible = e.target.checked;
    }
    renderLayers();
    requestRender();
  });

  $('addStitch').innerHTML = stitchOptionsHTML('satin', false);
  $('addLayerBtn').addEventListener('click', function () {
    if (!needImage()) return;
    if (layers.length >= MAX_LAYERS) { setStatus('Too many layers.'); return; }
    pushUndo(null);
    var L = makeLayer($('addStitch').value);
    layers.push(L);
    setActive(L.uid);
    setStatus('Added layer "' + L.id + '". The tools now paint into it.');
    requestRender();
  });

  $('lpName').addEventListener('change', function () {
    var L = activeLayer();
    if (!L) return;
    var id = uniqueId($('lpName').value, L);
    L.id = id;
    renderLayers();
  });
  $('lpCol').addEventListener('input', function () {
    var L = activeLayer();
    if (!L) return;
    L.colour = $('lpCol').value.toLowerCase();
    var dot = document.querySelector('.layer-row.active .cdot');
    if (dot) dot.style.background = L.colour;
    requestRender();
  });
  function bindParam(inputId, outId, key) {
    $(inputId).addEventListener('input', function () {
      var L = activeLayer();
      $(outId).textContent = $(inputId).value;
      if (!L) return;
      L[key] = parseFloat($(inputId).value);
      requestRender();
    });
  }
  bindParam('lpThread', 'lpThreadOut', 'thread');
  bindParam('lpAngle', 'lpAngleOut', 'angle');
  bindParam('lpKnotR', 'lpKnotROut', 'knot_r');
  bindParam('lpCoil', 'lpCoilOut', 'coil');

  function moveActive(dir) {
    var L = activeLayer();
    if (!L) return;
    var i = layers.indexOf(L), j = i + dir;
    if (j < 0 || j >= layers.length) return;
    pushUndo(null);
    layers = layers.slice();
    layers[i] = layers[j]; layers[j] = L;
    renderLayers();
    requestRender();
  }
  $('lpUp').addEventListener('click', function () { moveActive(1); });
  $('lpDown').addEventListener('click', function () { moveActive(-1); });
  $('lpDelete').addEventListener('click', function () {
    var L = activeLayer();
    if (!L) return;
    pushUndo(null);
    var i = layers.indexOf(L);
    layers = layers.filter(function (x) { return x !== L; });
    var next = layers[Math.min(i, layers.length - 1)];
    activeUid = next ? next.uid : 0;
    setStatus('Deleted layer "' + L.id + '" (Undo brings it back).');
    renderLayers();
    requestRender();
  });

  // ---------- Custom stitches ----------
  function renderCustom() {
    var list = $('customList');
    list.innerHTML = '';
    customStitches.forEach(function (c) {
      var li = document.createElement('li');
      var txt = document.createElement('span');
      txt.textContent = c.name + ' · ' + c.trigger + ' · ' + INIT_LABELS[c.init] + ' init · ' + loraPath(c.lora);
      var del = document.createElement('button');
      del.className = 'btn small'; del.textContent = 'Remove';
      del.addEventListener('click', function () {
        customStitches = customStitches.filter(function (x) { return x !== c; });
        var n = 0;
        layers.forEach(function (L) { if (L.stitch === c.name) { L.stitch = 'none'; n++; } });
        setStatus('Removed custom stitch "' + c.name + '"' + (n ? '; ' + n + ' layer(s) set to None.' : '.'));
        refreshStitchMenus();
        requestRender();
      });
      li.appendChild(txt); li.appendChild(del);
      list.appendChild(li);
    });
  }
  function refreshStitchMenus() {
    var prev = $('addStitch').value;
    $('addStitch').innerHTML = stitchOptionsHTML(prev, false);
    if (!$('addStitch').value) $('addStitch').value = 'satin';
    renderCustom();
    renderLayers();
  }
  function loraPath(file) {
    return /[\/\\:]/.test(file) ? file : 'loras/' + file;
  }
  $('csName').addEventListener('input', function () {
    $('csLora').placeholder = (slug($('csName').value) || 'my_stitch') + '.safetensors';
  });
  $('csAdd').addEventListener('click', function () {
    var name = $('csName').value.trim();
    var trig = $('csTrig').value.trim();
    var lora = $('csLora').value.trim() || (name + '.safetensors');
    if (!/^[a-z0-9_]+$/.test(name)) { setStatus('Stitch name: lower-case letters, digits and _ only.'); return; }
    if (BUILTIN.indexOf(name) >= 0 || name === 'none' || findCustom(name)) { setStatus('There is already a stitch called "' + name + '".'); return; }
    if (!trig) { setStatus('Enter the trigger word the LoRA was trained with.'); return; }
    if (/[\/\\:]/.test(lora)) { setStatus('LoRA file: just the file name, e.g. ' + name + '.safetensors'); return; }
    customStitches.push({ name: name, trigger: trig, init: $('csInit').value, lora: lora, extra: null });
    $('csName').value = ''; $('csTrig').value = ''; $('csLora').value = '';
    refreshStitchMenus();
    $('addStitch').value = name;
    setStatus('Added custom stitch "' + name + '". Pick it for a layer, or add a new layer with it.');
  });

  // ---------- Image loading ----------
  function setStatus(t) { $('status').textContent = t || ''; }

  function loadFromImage(img, name, opts) {
    opts = opts || {};
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
      return false;
    }
    outData = vctx.createImageData(W, H);
    ownerBuf = null;
    quant = null; lastQGroup = 0;
    layers = [];
    undoStack = [];
    if (!opts.noDefaultLayers) {
      BUILTIN.forEach(function (s) { layers.push(makeLayer(s)); });
      activeUid = layers[0].uid;
    } else activeUid = 0;
    importedSettings = null; importedOrder = null;
    stem = (name || 'image').replace(/\.[^.]*$/, '').replace(/[^A-Za-z0-9_\-]+/g, '_') || 'image';
    stage.classList.remove('empty');
    $('imgInfo').textContent = 'Loaded "' + (name || 'image') + '" at ' + W + ' x ' + H + ' px (from ' + w + ' x ' + h + '). Job name: ' + stem;
    setMinAreaRange();
    setStatus('');
    renderLayers();
    requestRender();
    return true;
  }

  function loadFile(file) {
    if (file && /\.zip$/i.test(file.name)) { openBundleFile(file); return; }
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
  // Each entry keeps the layer list as it was (so add/delete/move/quantise undo) and,
  // for paint actions, a copy of the one mask about to change.
  function pushUndo(maskLayer) {
    if (!baseData) return;
    undoStack.push({ list: layers.slice(), active: activeUid,
                     layer: maskLayer || null, mask: maskLayer ? maskLayer.mask.slice() : null });
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  }
  function undo() {
    if (!undoStack.length) { setStatus('Nothing to undo.'); return; }
    var e = undoStack.pop();
    layers = e.list;
    if (e.layer) e.layer.mask = e.mask;
    activeUid = findLayer(e.active) ? e.active : (layers.length ? layers[layers.length - 1].uid : 0);
    setStatus('');
    renderLayers();
    requestRender();
  }
  $('undoBtn').addEventListener('click', undo);

  // ---------- Tool selection ----------
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
    var L = activeLayer();
    if (!baseData || !L) return;
    pushUndo(L);
    L.mask.fill(0);
    requestRender();
  });

  function bindOut(inputId, outId) {
    $(inputId).addEventListener('input', function () { $(outId).textContent = $(inputId).value; });
  }
  bindOut('brushR', 'brushOut');
  bindOut('tol', 'tolOut');
  bindOut('qN', 'qNOut');
  bindOut('qSmooth', 'qSmoothOut');
  ['seed', 'previewToggle'].forEach(function (id) {
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
    var step = Math.max(0.5, rad / 3);
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
  var drag = null; // {tool, x0, y0, lx, ly, v, layer}
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

  // Returns the layer the tools paint into, or null (with a status message).
  function paintTarget() {
    if (quant) { setStatus('Create or discard the quantise preview first.'); return null; }
    var L = activeLayer();
    if (!L) { setStatus('Add a layer first (section 2).'); return null; }
    if (!L.visible) setStatus('Note: "' + L.id + '" is hidden, so you will not see this.');
    return L;
  }

  view.addEventListener('pointerdown', function (e) {
    if (!baseData || e.button !== 0) return;
    e.preventDefault();
    var L = paintTarget();
    if (!L) return;
    var p = toImage(e), tool = currentTool(), v = subtracting(e) ? 0 : 1;
    if (tool === 'wand') {
      var sx = clip(Math.floor(p.x), 0, W - 1), sy = clip(Math.floor(p.y), 0, H - 1);
      var t0 = performance.now();
      var mm = matchMap(sx, sy, parseFloat($('tol').value));
      var sel = $('nonContig').checked ? mm : floodFill(mm, sx, sy);
      pushUndo(L);
      applySelection(L.mask, sel, v);
      setStatus('Magic wand: ' + (performance.now() - t0).toFixed(0) + ' ms');
      requestRender();
      return;
    }
    pushUndo(L);
    view.setPointerCapture(e.pointerId);
    drag = { tool: tool, x0: p.x, y0: p.y, lx: p.x, ly: p.y, v: v, layer: L };
    if (tool === 'brush') {
      stampCircle(L.mask, p.x, p.y, parseInt($('brushR').value, 10), v);
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
        strokeLine(drag.layer.mask, drag.lx, drag.ly, p.x, p.y, parseInt($('brushR').value, 10), drag.v);
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
      fillRect(drag.layer.mask, drag.x0, drag.y0, p.x, p.y, drag.v);
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

  // ---------- Quantise by colours ----------
  var LIN = (function () {
    var t = new Float32Array(256);
    for (var i = 0; i < 256; i++) { var c = i / 255; t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
    return t;
  })();
  function labF(t) { return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116; }
  // CIE Lab (D65) of colour bins with 6 bits per channel (bin = r>>2, g>>2, b>>2), filled
  // lazily, so each distinct colour is converted once. NaN = not computed yet.
  var labBins = null;
  function pixelBin(d, q) { return ((d[q] >> 2) << 12) | ((d[q + 1] >> 2) << 6) | (d[q + 2] >> 2); }
  function binLab(bin) {
    if (!labBins) labBins = new Float32Array(262144 * 3).fill(NaN);
    var o = bin * 3;
    if (labBins[o] !== labBins[o]) {
      var r = LIN[((bin >> 12) << 2) + 2], g = LIN[(((bin >> 6) & 63) << 2) + 2], b = LIN[((bin & 63) << 2) + 2];
      var fx = labF((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047);
      var fy = labF(0.2126 * r + 0.7152 * g + 0.0722 * b);
      var fz = labF((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
      labBins[o] = 116 * fy - 16; labBins[o + 1] = 500 * (fx - fy); labBins[o + 2] = 200 * (fy - fz);
    }
    return o;
  }

  // k-means++ then Lloyd iterations on sampled Lab values (3 floats per sample).
  function kmeans(lab, k, rng) {
    var m = lab.length / 3, C = new Float32Array(k * 3), D = new Float64Array(m);
    var j, c, p, dl, da, db, dd;
    p = Math.floor(rng() * m) * 3;
    C[0] = lab[p]; C[1] = lab[p + 1]; C[2] = lab[p + 2];
    for (j = 0; j < m; j++) {
      p = j * 3; dl = lab[p] - C[0]; da = lab[p + 1] - C[1]; db = lab[p + 2] - C[2];
      D[j] = dl * dl + da * da + db * db;
    }
    for (c = 1; c < k; c++) {
      var sum = 0;
      for (j = 0; j < m; j++) sum += D[j];
      var pick = m - 1;
      if (sum > 0) {
        var r = rng() * sum, acc = 0;
        for (j = 0; j < m; j++) { acc += D[j]; if (acc >= r) { pick = j; break; } }
      } else pick = Math.floor(rng() * m);
      p = pick * 3;
      C[c * 3] = lab[p]; C[c * 3 + 1] = lab[p + 1]; C[c * 3 + 2] = lab[p + 2];
      for (j = 0; j < m; j++) {
        p = j * 3; dl = lab[p] - C[c * 3]; da = lab[p + 1] - C[c * 3 + 1]; db = lab[p + 2] - C[c * 3 + 2];
        dd = dl * dl + da * da + db * db;
        if (dd < D[j]) D[j] = dd;
      }
    }
    var asg = new Uint8Array(m).fill(255), sums = new Float64Array(k * 3), cnt = new Int32Array(k);
    for (var it = 0; it < 30; it++) {
      var changed = 0;
      sums.fill(0); cnt.fill(0);
      for (j = 0; j < m; j++) {
        p = j * 3;
        var L0 = lab[p], A0 = lab[p + 1], B0 = lab[p + 2], best = 0, bd = Infinity;
        for (c = 0; c < k; c++) {
          dl = L0 - C[c * 3]; da = A0 - C[c * 3 + 1]; db = B0 - C[c * 3 + 2];
          dd = dl * dl + da * da + db * db;
          if (dd < bd) { bd = dd; best = c; }
        }
        if (asg[j] !== best) { asg[j] = best; changed++; }
        sums[best * 3] += L0; sums[best * 3 + 1] += A0; sums[best * 3 + 2] += B0; cnt[best]++;
      }
      for (c = 0; c < k; c++) if (cnt[c]) {
        C[c * 3] = sums[c * 3] / cnt[c]; C[c * 3 + 1] = sums[c * 3 + 1] / cnt[c]; C[c * 3 + 2] = sums[c * 3 + 2] / cnt[c];
      }
      if (changed <= m * 0.001) break;
    }
    return C;
  }

  // Majority (mode) filter over a (2r+1)^2 window with a sliding histogram per row.
  // Label 255 = outside the clip; it is neither counted nor changed.
  function modeFilter(labels, k, r) {
    var out = labels.slice(), hist = new Int32Array(k);
    for (var y = 0; y < H; y++) {
      var y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r), row = y * W, x, yy, l;
      hist.fill(0);
      for (x = 0; x <= Math.min(W - 1, r); x++) {
        for (yy = y0; yy <= y1; yy++) { l = labels[yy * W + x]; if (l !== 255) hist[l]++; }
      }
      for (x = 0; x < W; x++) {
        var own = labels[row + x];
        if (own !== 255) {
          var best = own, bc = hist[own];
          for (var c = 0; c < k; c++) if (hist[c] > bc) { bc = hist[c]; best = c; }
          out[row + x] = best;
        }
        var xa = x + r + 1, xr = x - r;
        if (xa < W) for (yy = y0; yy <= y1; yy++) { l = labels[yy * W + xa]; if (l !== 255) hist[l]++; }
        if (xr >= 0) for (yy = y0; yy <= y1; yy++) { l = labels[yy * W + xr]; if (l !== 255) hist[l]--; }
      }
    }
    return out;
  }

  // Merge connected regions smaller than minArea into the neighbouring cluster that
  // shares the longest border with them; repeat until nothing changes.
  // Pass 1 labels every connected component once. A small component whose neighbour has
  // already merged into its colour is re-checked in later passes by a local flood fill on
  // the current labels, so the whole image is only scanned once.
  function absorbSmall(labels, k, minArea) {
    var n = W * H, comp = new Int32Array(n).fill(-1), queue = new Int32Array(n), border = new Int32Array(k);
    var starts = [], lens = [], qpos = 0, nc = 0, s, i, x, q, e, lab0;
    for (s = 0; s < n; s++) {
      if (labels[s] === 255 || comp[s] >= 0) continue;
      lab0 = labels[s];
      var start = qpos, head = qpos;
      comp[s] = nc; queue[qpos++] = s;
      while (head < qpos) {
        i = queue[head++]; x = i - ((i / W) | 0) * W;
        if (x > 0 && comp[i - 1] < 0 && labels[i - 1] === lab0) { comp[i - 1] = nc; queue[qpos++] = i - 1; }
        if (x < W - 1 && comp[i + 1] < 0 && labels[i + 1] === lab0) { comp[i + 1] = nc; queue[qpos++] = i + 1; }
        if (i >= W && comp[i - W] < 0 && labels[i - W] === lab0) { comp[i - W] = nc; queue[qpos++] = i - W; }
        if (i < n - W && comp[i + W] < 0 && labels[i + W] === lab0) { comp[i + W] = nc; queue[qpos++] = i + W; }
      }
      starts.push(start); lens.push(qpos - start); nc++;
    }
    // Longest-border neighbour cluster of the pixels queue[st .. st+ln), whose component id is cc.
    // Returns -2 if a neighbour already has the same colour (component grew), -1 if none.
    function bestNeighbour(st, ln, cc, own) {
      border.fill(0);
      for (var e2 = st; e2 < st + ln; e2++) {
        var p = queue[e2], px = p - ((p / W) | 0) * W;
        for (var dir = 0; dir < 4; dir++) {
          if (dir === 0) { if (px === 0) continue; q = p - 1; }
          else if (dir === 1) { if (px === W - 1) continue; q = p + 1; }
          else if (dir === 2) { if (p < W) continue; q = p - W; }
          else { if (p >= n - W) continue; q = p + W; }
          if (comp[q] === cc) continue;
          var lq = labels[q];
          if (lq === 255) continue;
          if (lq === own) return -2;
          border[lq]++;
        }
      }
      var best = -1, bc = 0;
      for (var b = 0; b < k; b++) if (border[b] > bc) { bc = border[b]; best = b; }
      return best;
    }
    var small = [];
    for (var c = 0; c < nc; c++) if (lens[c] < minArea) small.push(c);
    small.sort(function (a, b) { return lens[a] - lens[b]; });
    var pending = [], passes = 1;
    for (var si = 0; si < small.length; si++) {
      var cc = small[si], st = starts[cc], ln = lens[cc], own = labels[queue[st]];
      var best = bestNeighbour(st, ln, cc, own);
      if (best === -2) { pending.push(queue[st]); continue; }
      if (best < 0) continue;
      for (e = st; e < st + ln; e++) labels[queue[e]] = best;
    }
    // Later passes: flood fill from each pending seed on the current labels (component ids
    // from nc upwards, so they never clash with pass 1), stopping once it reaches minArea.
    var nextComp = nc;
    while (pending.length && passes < 12) {
      passes++;
      var again = [];
      for (var pi = 0; pi < pending.length; pi++) {
        s = pending[pi];
        lab0 = labels[s];
        var id = nextComp++, qs = 0, qh = 0, big = false;
        comp[s] = id; queue[qs++] = s;
        while (qh < qs) {
          if (qs >= minArea) { big = true; break; }
          i = queue[qh++]; x = i - ((i / W) | 0) * W;
          if (x > 0 && comp[i - 1] !== id && labels[i - 1] === lab0) { comp[i - 1] = id; queue[qs++] = i - 1; }
          if (x < W - 1 && comp[i + 1] !== id && labels[i + 1] === lab0) { comp[i + 1] = id; queue[qs++] = i + 1; }
          if (i >= W && comp[i - W] !== id && labels[i - W] === lab0) { comp[i - W] = id; queue[qs++] = i - W; }
          if (i < n - W && comp[i + W] !== id && labels[i + W] === lab0) { comp[i + W] = id; queue[qs++] = i + W; }
        }
        if (big || qs >= minArea) continue;
        var bn = bestNeighbour(0, qs, id, lab0);
        if (bn === -2) { again.push(s); continue; } // cannot happen for a complete fill; kept for safety
        if (bn < 0) continue;
        for (e = 0; e < qs; e++) labels[queue[e]] = bn;
      }
      pending = again;
    }
    return passes;
  }

  function setMinAreaRange() {
    var n = W * H;
    $('qMin').max = Math.max(10, Math.round(n * 0.005));
    $('qMin').value = Math.round(n * 0.0005);
    updateMinOut();
  }
  function updateMinOut() {
    var v = parseInt($('qMin').value, 10) || 0;
    $('qMinOut').textContent = v + ' px' + (W ? ' (' + (100 * v / (W * H)).toFixed(2) + '%)' : '');
  }
  $('qMin').addEventListener('input', updateMinOut);

  // Runs the full quantise; returns {labels, k, rgb, counts, clipUid, ms, timings}.
  function quantise(k, radius, minArea, clipUid) {
    var t0 = performance.now(), T = {};
    var n = W * H, clipL = clipUid ? findLayer(clipUid) : null, clipM = clipL ? clipL.mask : null;
    var d = baseData.data;
    // Downsampled copy: a grid of about 20k pixels (inside the clip) for k-means.
    var inside = 0, i, o;
    if (clipM) { for (i = 0; i < n; i++) inside += clipM[i]; } else inside = n;
    if (!inside) return null;
    var stride = Math.max(1, Math.floor(Math.sqrt(inside / 20000)));
    var sample = [];
    for (var y = 0; y < H; y += stride) for (var x = 0; x < W; x += stride) {
      i = y * W + x;
      if (!clipM || clipM[i]) sample.push(i);
    }
    if (sample.length < k) { sample = []; for (i = 0; i < n; i++) if (!clipM || clipM[i]) sample.push(i); }
    var slab = new Float32Array(sample.length * 3);
    for (var j = 0; j < sample.length; j++) {
      o = binLab(pixelBin(d, sample[j] * 4));
      slab[j * 3] = labBins[o]; slab[j * 3 + 1] = labBins[o + 1]; slab[j * 3 + 2] = labBins[o + 2];
    }
    var C = kmeans(slab, k, mulberry32(1));
    T.kmeans = performance.now() - t0;
    // Assign every pixel: nearest centre in Lab, worked out once per colour bin.
    var labels = new Uint8Array(n).fill(255), binLabel = new Uint8Array(262144).fill(255);
    for (i = 0; i < n; i++) {
      if (clipM && !clipM[i]) continue;
      var bin = pixelBin(d, i * 4), best = binLabel[bin];
      if (best === 255) {
        o = binLab(bin);
        var L0 = labBins[o], A0 = labBins[o + 1], B0 = labBins[o + 2], bd = Infinity;
        best = 0;
        for (var c = 0; c < k; c++) {
          var dl = L0 - C[c * 3], da = A0 - C[c * 3 + 1], db = B0 - C[c * 3 + 2];
          var dd = dl * dl + da * da + db * db;
          if (dd < bd) { bd = dd; best = c; }
        }
        binLabel[bin] = best;
      }
      labels[i] = best;
    }
    var t1 = performance.now();
    T.assign = t1 - t0 - T.kmeans;
    if (radius > 0) labels = modeFilter(labels, k, radius);
    var t2 = performance.now(); T.smooth = t2 - t1;
    var passes = minArea > 0 ? absorbSmall(labels, k, minArea) : 0;
    var t3 = performance.now(); T.merge = t3 - t2; T.mergePasses = passes;
    // Cluster mean colours in sRGB from the final labels.
    var sums = new Float64Array(k * 3), counts = new Int32Array(k);
    for (i = 0; i < n; i++) {
      var l = labels[i];
      if (l === 255) continue;
      sums[l * 3] += d[i * 4]; sums[l * 3 + 1] += d[i * 4 + 1]; sums[l * 3 + 2] += d[i * 4 + 2]; counts[l]++;
    }
    var rgb = new Uint8ClampedArray(k * 3);
    for (c = 0; c < k; c++) if (counts[c]) {
      rgb[c * 3] = sums[c * 3] / counts[c]; rgb[c * 3 + 1] = sums[c * 3 + 1] / counts[c]; rgb[c * 3 + 2] = sums[c * 3 + 2] / counts[c];
    }
    var ms = performance.now() - t0;
    return { labels: labels, k: k, rgb: rgb, counts: counts, clipUid: clipUid, ms: ms, timings: T, params: [k, radius, minArea, clipUid].join('|') };
  }

  function quantParams() {
    return { k: parseInt($('qN').value, 10), r: parseInt($('qSmooth').value, 10),
             min: parseInt($('qMin').value, 10) || 0, clip: parseInt($('qClip').value, 10) || 0 };
  }
  function runQuantise() {
    var q = quantParams();
    var res = quantise(q.k, q.r, q.min, q.clip);
    if (!res) { setStatus('The chosen clip layer is empty.'); return null; }
    var used = 0;
    for (var c = 0; c < res.k; c++) if (res.counts[c]) used++;
    quant = res;
    setStatus('Quantised into ' + used + ' colours in ' + res.ms.toFixed(0) + ' ms. Create layers to keep it, or change the settings and preview again.');
    requestRender();
    return res;
  }
  function withBusy(msg, fn) {
    setStatus(msg);
    setTimeout(fn, 20); // let the status paint first
  }
  $('qPreviewBtn').addEventListener('click', function () {
    if (!needImage()) return;
    withBusy('Quantising…', runQuantise);
  });
  $('qDiscardBtn').addEventListener('click', function () {
    quant = null; setStatus(''); requestRender();
  });
  function commitQuantise() {
    var res = quant;
    var q = quantParams();
    if (!res || res.params !== [q.k, q.r, q.min, q.clip].join('|')) res = runQuantise();
    if (!res) return;
    pushUndo(null);
    var removed = 0;
    if ($('qReplace').checked && lastQGroup) {
      var before = layers.length;
      layers = layers.filter(function (L) { return L.qgroup !== lastQGroup; });
      removed = before - layers.length;
    }
    var group = ++qCounter;
    var order = [];
    for (var c = 0; c < res.k; c++) if (res.counts[c]) order.push(c);
    order.sort(function (a, b) { return res.counts[b] - res.counts[a]; }); // largest paints first
    if (layers.length + order.length > MAX_LAYERS) { setStatus('Too many layers.'); undoStack.pop(); return; }
    var n = W * H, labels = res.labels;
    order.forEach(function (c) {
      var m = new Uint8Array(n);
      for (var i = 0; i < n; i++) if (labels[i] === c) m[i] = 1;
      var L = makeLayer('none', { id: nextId('colour'), mask: m, qgroup: group,
                                  colour: rgbToHex(res.rgb[c * 3], res.rgb[c * 3 + 1], res.rgb[c * 3 + 2]) });
      layers.push(L);
    });
    lastQGroup = group;
    quant = null;
    activeUid = layers.length ? layers[layers.length - 1].uid : 0;
    renderLayers();
    requestRender();
    setStatus('Created ' + order.length + ' colour layers' + (removed ? ' (replaced ' + removed + ' from the last quantise)' : '') +
              ' in ' + res.ms.toFixed(0) + ' ms. Pick a stitch for each one; layers left as None are not exported.');
  }
  $('qApplyBtn').addEventListener('click', function () {
    if (!needImage()) return;
    withBusy('Quantising…', commitQuantise);
  });

  // ---------- Export helpers ----------
  function downloadBlob(blob, filename) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  }
  function downloadCanvas(canvas, filename) {
    canvas.toBlob(function (blob) { downloadBlob(blob, filename); }, 'image/png');
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

  // ---------- CRC-32, zip (store only) and greyscale PNG ----------
  var CRC_TABLE = (function () {
    var t = new Uint32Array(256);
    for (var n = 0; n < 256; n++) {
      var c = n;
      for (var k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      t[n] = c >>> 0;
    }
    return t;
  })();
  function crc32(u8, crc) {
    var c = (crc === undefined ? 0 : crc) ^ 0xFFFFFFFF;
    for (var i = 0; i < u8.length; i++) c = CRC_TABLE[(c ^ u8[i]) & 255] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }
  function utf8(s) { return new TextEncoder().encode(s); }

  // files: [{name, data: Uint8Array}] -> Blob (application/zip, no compression).
  function makeZip(files) {
    var now = new Date();
    var dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
    var dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    var parts = [], central = [], offset = 0;
    files.forEach(function (f) {
      var name = utf8(f.name), crc = crc32(f.data), size = f.data.length;
      var lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true);
      lh.setUint16(8, 0, true); lh.setUint16(10, dosTime, true); lh.setUint16(12, dosDate, true);
      lh.setUint32(14, crc, true); lh.setUint32(18, size, true); lh.setUint32(22, size, true);
      lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
      parts.push(new Uint8Array(lh.buffer), name, f.data);
      var ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true);
      ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true); ch.setUint16(12, dosTime, true);
      ch.setUint16(14, dosDate, true); ch.setUint32(16, crc, true); ch.setUint32(20, size, true);
      ch.setUint32(24, size, true); ch.setUint16(28, name.length, true);
      ch.setUint32(42, offset, true);
      central.push(new Uint8Array(ch.buffer), name);
      offset += 30 + name.length + size;
    });
    var cdSize = 0;
    central.forEach(function (c) { cdSize += c.length; });
    var end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
    return new Blob(parts.concat(central, [new Uint8Array(end.buffer)]), { type: 'application/zip' });
  }

  // Reads a zip (stored, or deflated where DecompressionStream exists) -> Promise of {name: Uint8Array}.
  function readZip(buf) {
    var u8 = new Uint8Array(buf), dv = new DataView(buf);
    var eocd = -1;
    for (var p = u8.length - 22; p >= Math.max(0, u8.length - 65557); p--) {
      if (dv.getUint32(p, true) === 0x06054b50) { eocd = p; break; }
    }
    if (eocd < 0) return Promise.reject(new Error('not a zip file'));
    var count = dv.getUint16(eocd + 10, true), off = dv.getUint32(eocd + 16, true);
    var jobs = [], out = {};
    var dec = new TextDecoder();
    for (var e = 0; e < count; e++) {
      if (dv.getUint32(off, true) !== 0x02014b50) return Promise.reject(new Error('bad zip directory'));
      var method = dv.getUint16(off + 10, true), csize = dv.getUint32(off + 20, true);
      var nlen = dv.getUint16(off + 28, true), xlen = dv.getUint16(off + 30, true), clen = dv.getUint16(off + 32, true);
      var loc = dv.getUint32(off + 42, true);
      var name = dec.decode(u8.subarray(off + 46, off + 46 + nlen));
      off += 46 + nlen + xlen + clen;
      if (/\/$/.test(name)) continue;
      var dstart = loc + 30 + dv.getUint16(loc + 26, true) + dv.getUint16(loc + 28, true);
      var data = u8.subarray(dstart, dstart + csize);
      if (method === 0) out[name] = data;
      else if (method === 8) {
        if (typeof DecompressionStream === 'undefined') return Promise.reject(new Error('this browser cannot read compressed zips'));
        jobs.push((function (nm, d) {
          var ds = new DecompressionStream('deflate-raw');
          return new Response(new Blob([d]).stream().pipeThrough(ds)).arrayBuffer().then(function (ab) { out[nm] = new Uint8Array(ab); });
        })(name, data));
      } else return Promise.reject(new Error('unsupported zip compression (method ' + method + ')'));
    }
    return Promise.all(jobs).then(function () { return out; });
  }

  // zlib stream of `raw`: real deflate via CompressionStream where available, else stored blocks.
  function zlibDeflate(raw) {
    if (typeof CompressionStream !== 'undefined') {
      try {
        var cs = new CompressionStream('deflate');
        return new Response(new Blob([raw]).stream().pipeThrough(cs)).arrayBuffer()
          .then(function (ab) { return new Uint8Array(ab); });
      } catch (e) { /* fall through */ }
    }
    var nb = Math.max(1, Math.ceil(raw.length / 65535));
    var out = new Uint8Array(2 + raw.length + nb * 5 + 4), o = 0;
    out[o++] = 0x78; out[o++] = 0x01;
    for (var b = 0; b < nb; b++) {
      var s = b * 65535, len = Math.min(65535, raw.length - s);
      out[o++] = b === nb - 1 ? 1 : 0;
      out[o++] = len & 255; out[o++] = len >> 8; out[o++] = ~len & 255; out[o++] = (~len >> 8) & 255;
      out.set(raw.subarray(s, s + len), o); o += len;
    }
    var a = 1, bb = 0;
    for (var i = 0; i < raw.length; i++) { a = (a + raw[i]) % 65521; bb = (bb + a) % 65521; }
    var ad = ((bb << 16) | a) >>> 0;
    out[o++] = ad >>> 24; out[o++] = (ad >>> 16) & 255; out[o++] = (ad >>> 8) & 255; out[o++] = ad & 255;
    return Promise.resolve(out);
  }
  function pngChunk(type, data) {
    var c = new Uint8Array(12 + data.length), dv = new DataView(c.buffer);
    dv.setUint32(0, data.length);
    for (var i = 0; i < 4; i++) c[4 + i] = type.charCodeAt(i);
    c.set(data, 8);
    dv.setUint32(8 + data.length, crc32(c.subarray(4, 8 + data.length)));
    return c;
  }
  // 8-bit greyscale PNG (colour type 0) from one byte per pixel.
  function encodeGreyPNG(grey, w, h) {
    var raw = new Uint8Array((w + 1) * h);
    for (var y = 0; y < h; y++) raw.set(grey.subarray(y * w, (y + 1) * w), y * (w + 1) + 1); // filter 0
    return zlibDeflate(raw).then(function (z) {
      var ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
      dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = 8; ihdr[9] = 0;
      var parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), pngChunk('IHDR', ihdr),
                   pngChunk('IDAT', z), pngChunk('IEND', new Uint8Array(0))];
      var len = 0; parts.forEach(function (p) { len += p.length; });
      var out = new Uint8Array(len), o = 0;
      parts.forEach(function (p) { out.set(p, o); o += p.length; });
      return out;
    });
  }
  function canvasPNGBytes(imgData) {
    return new Promise(function (resolve, reject) {
      canvasFrom(imgData).toBlob(function (b) {
        if (!b) { reject(new Error('could not encode the image')); return; }
        b.arrayBuffer().then(function (ab) { resolve(new Uint8Array(ab)); });
      }, 'image/png');
    });
  }

  // ---------- Job bundle ----------
  function exportedLayers() {
    return layers.filter(function (L) { return isExported(L); });
  }
  function jobSeed() { var s = parseInt($('jobSeed').value, 10); return isNaN(s) ? 1600 : s; }
  function jobVariants() { var v = parseInt($('jobVariants').value, 10); return isNaN(v) || v < 0 ? 0 : v; }

  // Builds job.json plus the resolved per-layer masks (each pixel in at most one layer).
  function buildJob() {
    var own = computeOwner(isExported);
    var out = [], used = [], skipped = [];
    for (var li = 0; li < layers.length; li++) {
      var L = layers[li];
      if (!isExported(L)) continue;
      var g = new Uint8Array(W * H), any = false, v = li + 1;
      for (var i = 0; i < g.length; i++) if (own[i] === v) { g[i] = 255; any = true; }
      if (!any) { skipped.push(L.id); continue; }
      var st = initStyle(L.stitch);
      var e = { id: L.id, stitch: L.stitch, mask: 'masks/' + L.id + '.png', colour: L.colour };
      if (st === 'satin') { e.angle = L.angle; e.thread = L.thread; }
      else if (st === 'french_knot') e.knot_r = L.knot_r;
      else if (st === 'silk_purl') e.coil = L.coil;
      if (L.extra) for (var key in L.extra) if (!(key in e)) e[key] = L.extra[key];
      out.push({ entry: e, grey: g });
      if (BUILTIN.indexOf(L.stitch) < 0 && used.indexOf(L.stitch) < 0) used.push(L.stitch);
    }
    var job = { format: 'reintegration-job', version: 1, name: stem, image: 'image.png',
                layers: out.map(function (o) { return o.entry; }) };
    if (used.length) {
      job.stitches = {};
      used.forEach(function (name) {
        var c = findCustom(name);
        var s = { lora: loraPath(c.lora), trigger: c.trigger, init: c.init };
        if (c.extra) for (var key in c.extra) if (!(key in s)) s[key] = c.extra[key];
        job.stitches[name] = s;
      });
    }
    job.order = importedOrder ? importedOrder.slice() : ['french_knot', 'silk_purl', 'satin'].concat(used);
    var settings = {}, k;
    for (k in RECIPE) settings[k] = RECIPE[k];
    if (importedSettings) for (k in importedSettings) settings[k] = importedSettings[k];
    settings.per_region = true;
    settings.region_variants = jobVariants();
    settings.seed = jobSeed();
    job.settings = settings;
    return { job: job, masks: out, skipped: skipped };
  }

  // Promise of {blob, job, skipped}. Also used by the tests via window.tryit.
  function buildBundle() {
    if (!baseData) return Promise.reject(new Error('no image loaded'));
    var b = buildJob();
    if (!b.masks.length) return Promise.reject(new Error('no layer with a stitch has any pixels; pick a stitch (not None) and paint a mask'));
    var files = [{ name: 'job.json', data: utf8(JSON.stringify(b.job, null, 2) + '\n') }];
    return canvasPNGBytes(baseData).then(function (img) {
      files.push({ name: 'image.png', data: img });
      return Promise.all(b.masks.map(function (m) { return encodeGreyPNG(m.grey, W, H); }));
    }).then(function (pngs) {
      pngs.forEach(function (p, i) { files.push({ name: b.masks[i].entry.mask, data: p }); });
      return { blob: makeZip(files), job: b.job, skipped: b.skipped };
    });
  }

  function saveBundle() {
    if (!needImage()) return Promise.resolve(null);
    setStatus('Building job bundle…');
    return buildBundle().then(function (r) {
      var fn = stem + '.zip';
      downloadBlob(r.blob, fn);
      var msg = 'Saved ' + fn + ' with ' + r.job.layers.length + ' layer(s), ' + (r.blob.size / 1024).toFixed(0) + ' KB.';
      if (r.skipped.length) msg += ' Skipped (no pixels left after overlaps): ' + r.skipped.join(', ') + '.';
      setStatus(msg);
      return r;
    }).catch(function (e) { setStatus('Could not build the bundle: ' + e.message + '.'); return null; });
  }
  $('dlBundle').addEventListener('click', saveBundle);
  $('colabBtn').addEventListener('click', function () {
    if (!needImage()) return;
    if (!exportedLayers().length) { setStatus('Pick a stitch (not None) for at least one layer first.'); return; }
    window.open(COLAB_URL, '_blank', 'noopener'); // opened now, inside the click, so it is not blocked
    saveBundle();
  });

  // ---------- Open a job bundle ----------
  function decodeImage(bytes, type) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(new Blob([bytes], { type: type || 'image/png' }));
      var img = new Image();
      img.onload = function () { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = function () { URL.revokeObjectURL(url); reject(new Error('could not decode an image in the bundle')); };
      img.src = url;
    });
  }
  // Mask image -> Uint8Array(W*H), white (>=128) = 1, scaled to the working size if needed.
  function maskFromImage(img) {
    var c = document.createElement('canvas');
    c.width = W; c.height = H;
    var cx = c.getContext('2d');
    cx.imageSmoothingEnabled = false;
    cx.drawImage(img, 0, 0, W, H);
    var d = cx.getImageData(0, 0, W, H).data, m = new Uint8Array(W * H);
    for (var i = 0, k = 0; i < m.length; i++, k += 4) m[i] = d[k] >= 128 ? 1 : 0;
    return m;
  }
  function splitKnown(obj, keys) {
    var extra = null;
    for (var key in obj) if (keys.indexOf(key) < 0) { extra = extra || {}; extra[key] = obj[key]; }
    return extra;
  }

  // Promise; restores image, layers, custom stitches and settings from zip bytes.
  function importBundle(buf) {
    var files, job, prefix = '';
    return readZip(buf).then(function (f) {
      files = f;
      var jp = null;
      Object.keys(files).forEach(function (n) {
        if (/(^|\/)job\.json$/.test(n) && (jp === null || n.length < jp.length)) jp = n;
      });
      if (jp === null) throw new Error('no job.json in the zip');
      prefix = jp.slice(0, jp.length - 'job.json'.length);
      job = JSON.parse(new TextDecoder().decode(files[jp]));
      if (job.format !== 'reintegration-job') throw new Error('job.json is not a reintegration job');
      var imgName = prefix + (job.image || 'image.png');
      if (!files[imgName]) throw new Error('missing ' + imgName);
      return decodeImage(files[imgName], /\.jpe?g$/i.test(imgName) ? 'image/jpeg' : 'image/png');
    }).then(function (img) {
      if (!loadFromImage(img, job.name || 'job', { noDefaultLayers: true })) throw new Error('could not read the image');
      // Custom stitches.
      customStitches = [];
      var st = job.stitches || {};
      Object.keys(st).forEach(function (name) {
        var s = st[name] || {};
        if (!/^[a-z0-9_]+$/.test(name) || BUILTIN.indexOf(name) >= 0) return;
        var lora = String(s.lora || name + '.safetensors').replace(/^loras\//, '');
        customStitches.push({ name: name, trigger: String(s.trigger || name),
                              init: ['satin', 'french_knot', 'silk_purl', 'flat'].indexOf(s.init) >= 0 ? s.init : 'flat',
                              lora: lora, extra: splitKnown(s, STITCH_KEYS) });
      });
      var entries = Array.isArray(job.layers) ? job.layers : [];
      return Promise.all(entries.map(function (e) {
        var mp = prefix + (e.mask || ('masks/' + e.id + '.png'));
        if (!files[mp]) return Promise.resolve(null);
        return decodeImage(files[mp]).then(maskFromImage);
      })).then(function (masks) {
        var warn = [];
        entries.forEach(function (e, i) {
          if (!masks[i]) { warn.push('no mask for ' + e.id); return; }
          var stitch = e.stitch;
          if (BUILTIN.indexOf(stitch) < 0 && !findCustom(stitch)) { warn.push('unknown stitch ' + stitch); stitch = 'none'; }
          var L = makeLayer(stitch, { mask: masks[i], colour: isHex(e.colour) ? e.colour.toLowerCase() : undefined,
                                      extra: splitKnown(e, LAYER_KEYS) });
          L.id = uniqueId(e.id || stitch);
          if (+e.angle) L.angle = +e.angle;
          if (+e.thread) L.thread = +e.thread;
          if (+e.knot_r) L.knot_r = +e.knot_r;
          if (+e.coil) L.coil = +e.coil;
          layers.push(L);
        });
        var s = job.settings || {};
        importedSettings = {};
        for (var k in s) if (k !== 'seed' && k !== 'region_variants') importedSettings[k] = s[k];
        importedOrder = Array.isArray(job.order) ? job.order.slice() : null;
        if (s.seed !== undefined) $('jobSeed').value = s.seed;
        if (s.region_variants !== undefined) $('jobVariants').value = s.region_variants;
        activeUid = layers.length ? layers[layers.length - 1].uid : 0;
        refreshStitchMenus();
        requestRender();
        var loras = Object.keys(files).filter(function (n) { return n.indexOf(prefix + 'loras/') === 0; }).length;
        var msg = 'Opened job "' + stem + '": ' + layers.length + ' layer(s), ' + customStitches.length + ' custom stitch(es).';
        if (loras) msg += ' LoRA files in the zip are not kept; add them again when you run it.';
        if (warn.length) msg += ' Warnings: ' + warn.join('; ') + '.';
        setStatus(msg);
        return { job: job, warnings: warn };
      });
    });
  }
  function openBundleFile(file) {
    setStatus('Opening ' + file.name + '…');
    file.arrayBuffer().then(importBundle).catch(function (e) { setStatus('Could not open the bundle: ' + e.message + '.'); });
  }
  $('bundleInput').addEventListener('change', function (e) {
    if (e.target.files && e.target.files[0]) openBundleFile(e.target.files[0]);
    e.target.value = '';
  });

  // ---------- Other downloads ----------
  $('dlImage').addEventListener('click', function () {
    if (!needImage()) return;
    downloadCanvas(canvasFrom(baseData), stem + '.png');
  });
  // Legacy: one mask per stitch, the union of that stitch's exported layers after overlaps.
  $('dlMasks').addEventListener('click', function () {
    if (!needImage()) return;
    var b = buildJob(), byStitch = {}, names = [];
    b.masks.forEach(function (m) {
      var s = m.entry.stitch;
      if (!byStitch[s]) { byStitch[s] = new Uint8Array(W * H); names.push(s); }
      var u = byStitch[s], g = m.grey;
      for (var i = 0; i < g.length; i++) if (g[i]) u[i] = 255;
    });
    var saved = [], delay = 0;
    names.forEach(function (s) {
      var id = new ImageData(W, H), d = id.data, u = byStitch[s];
      for (var i = 0, k = 0; i < u.length; i++, k += 4) { d[k] = u[i]; d[k + 1] = u[i]; d[k + 2] = u[i]; d[k + 3] = 255; }
      var name = stem + '__' + s + '.png';
      saved.push(name);
      (function (c, n, t) { setTimeout(function () { downloadCanvas(c, n); }, t); })(canvasFrom(id), name, delay);
      delay += 400; // spaced out so browsers accept multiple downloads
    });
    setStatus(saved.length ? 'Saving: ' + saved.join(', ') : 'No layer with a stitch has any pixels; nothing to save.');
  });
  $('dlPreview').addEventListener('click', function () {
    if (!needImage()) return;
    var id = composite(new ImageData(W, H), true);
    downloadCanvas(canvasFrom(id), stem + '__preview.png');
  });

  window.addEventListener('resize', drawUI);
  renderLayers();
  renderCustom();

  // Hooks for automated testing from the browser console.
  window.tryit = {
    buildBundle: buildBundle,
    readZip: function (blob) { return blob.arrayBuffer().then(readZip); },
    importBundle: function (blob) { return blob.arrayBuffer().then(importBundle); },
    quantise: quantise,
    state: function () { return { W: W, H: H, layers: layers, customStitches: customStitches, stem: stem }; }
  };
})();
