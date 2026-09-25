/* Hero widget: the loss strip (paper Fig. 8) as four animated steps. */
(function () {
  'use strict';
  var root = document.getElementById('hero');
  if (!root) return;
  var BASE = 'assets/strip/';

  // Secondary line under each step caption: the paper's Fig. 8 caption, part by part.
  var SUB = [
    'The maker’s design survives as inked underdrawing after stitch loss.',
    'The conservator denotes the stitch techniques with masks.',
    'Each region is initialised with a procedural stand-in of its prescribed stitch.',
    'Each region is denoised using its stitch LoRA, and the reintegration composed from per-region seed variants.'
  ];
  var SHORT = ['Photograph', 'Mask', 'Procedural fill', 'Denoise'];
  var DUR = [3500, 3800, 4000, 4400];   // ms per step
  var FADE = 700;                        // crossfade from the previous state at the start of a step
  var KEYS = ['m0', 'm1', 'm2', 'i0', 'i1', 'i2', 'fin'];

  RL.json(BASE + 'manifest.json').then(build).catch(function (e) {
    console.warn('hero:', e);   // keep the static fallback image
  });

  function sm(t, a, b) {
    var x = Math.min(1, Math.max(0, (t - a) / (b - a)));
    return x * x * (3 - 2 * x);
  }

  function build(m) {
    var steps = m.steps, masks = m.masks, inits = m.inits;
    root.innerHTML = '';

    var stage = RL.el('div', { class: 'hero-stage', role: 'img' });
    var layers = {};
    stage.appendChild(RL.el('img', { class: 'hero-base', src: BASE + steps[0].file, alt: '', width: 900, height: 900 }));
    function layer(key, file) {
      var img = RL.el('img', { src: BASE + file, alt: '', width: 900, height: 900, decoding: 'async', draggable: 'false' });
      img.style.opacity = 0;
      stage.appendChild(img);
      layers[key] = img;
    }
    // Masks sit above the (full-frame) procedural fills, so a region keeps its
    // mask colour until its own fill replaces it.
    inits.forEach(function (it, i) { layer('i' + i, it.file); });
    masks.forEach(function (mk, i) { layer('m' + i, mk.file); });
    layer('fin', steps[3].file);

    var playBtn = RL.el('button', { type: 'button', class: 'iconbtn' });
    var list = RL.el('ol', { class: 'hero-steps' });
    var btns = steps.map(function (s, i) {
      var b = RL.el('button', { type: 'button', 'aria-label': 'Step ' + (i + 1) + ': ' + s.caption },
        [RL.el('span', { class: 'num', text: String(i + 1) }), RL.el('span', { class: 'lbl', text: SHORT[i] }),
          RL.el('span', { class: 'bar', 'aria-hidden': 'true' })]);
      b.addEventListener('click', function () { setPlaying(false); go(i); });
      list.appendChild(RL.el('li', null, b));
      return b;
    });
    var caption = RL.el('p', { class: 'hero-caption' });
    var sub = RL.el('p', { class: 'hero-sub' });
    var legend = RL.el('ul', { class: 'hero-legend', 'aria-label': 'Stitch types' });
    var legendItems = masks.map(function (mk) {
      var li = RL.el('li', null, [RL.el('span', { class: 'sw', style: 'background:' + mk.colour }), mk.label]);
      legend.appendChild(li);
      return li;
    });

    root.appendChild(stage);
    root.appendChild(RL.el('div', { class: 'hero-controls' }, [playBtn, list]));
    root.appendChild(caption);
    root.appendChild(sub);
    root.appendChild(legend);

    var cur = {}; KEYS.forEach(function (k) { cur[k] = 0; });
    var snap = {}, step = 0, t = 0, playing = !RL.reducedMotion(), onScreen = true, raf = null, last = 0;
    var ready = false, lastSub = '';

    function target(s, tt) {
      var o = { m0: 0, m1: 0, m2: 0, i0: 0, i1: 0, i2: 0, fin: 0 }, wipe = 1, lit = [], line = SUB[s];
      if (s === 1) {
        o.m0 = sm(tt, 300, 900); o.m1 = sm(tt, 1100, 1700); o.m2 = sm(tt, 1900, 2500);
        if (tt >= 300) lit.push(0);
        if (tt >= 1100) lit.push(1);
        if (tt >= 1900) lit.push(2);
      } else if (s === 2) {
        o.i0 = sm(tt, 0, 700); o.i1 = sm(tt, 1300, 2000); o.i2 = sm(tt, 2600, 3300);
        o.m0 = 1 - o.i0; o.m1 = 1 - o.i1; o.m2 = 1 - o.i2;
        var k = tt < 1300 ? 0 : tt < 2600 ? 1 : 2;
        for (var j = 0; j <= k; j++) lit.push(j);
        line = SUB[2] + ' ' + inits[k].caption + '.';
      } else if (s === 3) {
        o.i2 = 1; o.fin = 1; wipe = sm(tt, 300, 2800);
      }
      return { o: o, wipe: wipe, lit: lit, line: line };
    }

    function render() {
      var tg = target(step, t), a = FADE ? sm(t, 0, FADE) : 1;
      KEYS.forEach(function (k) {
        var v = snap[k] + (tg.o[k] - snap[k]) * a;
        if (Math.abs(v - cur[k]) > 0.002 || (v === 0) !== (cur[k] === 0)) {
          layers[k].style.opacity = v.toFixed(3);
          cur[k] = v;
        }
      });
      var fin = layers.fin;
      if (step === 3 && tg.wipe < 1) {
        var soft = 18, x = tg.wipe * (100 + soft) - soft;
        var g = 'linear-gradient(90deg,#000 ' + x.toFixed(2) + '%,transparent ' + (x + soft).toFixed(2) + '%)';
        fin.style.webkitMaskImage = g; fin.style.maskImage = g;
      } else if (fin.style.maskImage || fin.style.webkitMaskImage) {
        fin.style.webkitMaskImage = ''; fin.style.maskImage = '';
      }
      legend.classList.toggle('hidden', step === 0 || step === 3);
      legendItems.forEach(function (li, i) { li.classList.toggle('on', tg.lit.indexOf(i) >= 0); });
      if (tg.line !== lastSub) { sub.textContent = tg.line; lastSub = tg.line; }
      btns[step].querySelector('.bar').style.width = playing ? (Math.min(1, t / DUR[step]) * 100).toFixed(1) + '%' : '0';
    }

    function go(i) {
      KEYS.forEach(function (k) { snap[k] = cur[k]; });
      btns.forEach(function (b, j) {
        if (j === i) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
        b.querySelector('.bar').style.width = '0';
      });
      step = i;
      caption.textContent = steps[i].caption;
      stage.setAttribute('aria-label', 'Step ' + (i + 1) + ' of 4: ' + steps[i].caption);
      if (RL.reducedMotion()) { t = DUR[i]; FADE = 0; } else { t = 0; FADE = i === 0 ? 900 : 700; }
      render();
      kick();
    }

    function frame(now) {
      raf = null;
      var dt = Math.min(100, now - last); last = now;
      if (t < DUR[step]) { t = Math.min(DUR[step], t + dt); render(); }
      else if (playing) { go((step + 1) % 4); return; }
      if (onScreen && (playing || t < DUR[step])) raf = requestAnimationFrame(frame);
    }
    function kick() {
      if (!raf && ready && onScreen && (playing || t < DUR[step])) {
        last = performance.now();
        raf = requestAnimationFrame(frame);
      }
    }
    function setPlaying(p) {
      playing = p;
      RL.setPlayButton(playBtn, playing, 'the four-step animation');
      if (!p) btns[step].querySelector('.bar').style.width = '0';
      kick();
    }
    playBtn.addEventListener('click', function () { setPlaying(!playing); });

    RL.setPlayButton(playBtn, playing, 'the four-step animation');
    go(0);
    RL.onVisibility(stage, function (v) { onScreen = v; kick(); });
    // Start once every layer has loaded so the first loop never shows half-loaded images.
    var urls = Object.keys(layers).map(function (k) { return layers[k].src; });
    RL.loadAll(urls).then(function () { ready = true; kick(); });
  }
})();
