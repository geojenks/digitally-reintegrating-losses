/* "Controllable structure": procedural init vs generated fill, one parameter swept per stop. */
(function () {
  'use strict';
  var root = document.getElementById('structure');
  if (!root) return;
  var BASE = 'assets/structure/';
  var STEP_MS = 2200, FADE = 250;

  // Short button labels, and the end of each caption line, per stop id.
  var SHORT = { knot: 'Knot radius', thread: 'Thread width', angle: 'Thread angle', coil: 'Coil pitch', colour: 'Colour' };
  var TINY = { knot: 'Knot', thread: 'Width', angle: 'Angle', coil: 'Coil', colour: 'Colour' };   // phones
  var CHANGES = { knot: 'only the radius changes', thread: 'only the thread width changes',
    angle: 'only the thread angle changes', coil: 'only the coil pitch changes', colour: 'only the init colours change' };
  var ICON = {
    left: '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M9.5 1.5 4 7l5.5 5.5 1.2-1.2L6.4 7l4.3-4.3z"/></svg>',
    right: '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M4.5 1.5 10 7l-5.5 5.5-1.2-1.2L7.6 7 3.3 2.7z"/></svg>',
    up: '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M1.5 9.5 7 4l5.5 5.5-1.2 1.2L7 6.4 2.7 10.7z"/></svg>',
    down: '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M1.5 4.5 7 10l5.5-5.5-1.2-1.2L7 7.6 2.7 3.3z"/></svg>'
  };

  RL.whenNear(root, function () {
    RL.json(BASE + 'manifest.json').then(build).catch(function (e) {
      root.innerHTML = '<p class="widget-loading">Could not load the parameter sweeps.</p>';
      console.warn('structure:', e);
    });
  });

  function iconBtn(icon, label) {
    return RL.el('button', { type: 'button', class: 'iconbtn', 'aria-label': label, title: label, html: icon });
  }
  function valueText(s, v) { return s.unit ? v.label + (s.unit === '°' ? '' : ' ') + s.unit : v.label; }

  function build(m) {
    var stops = m.stops;
    var state = { s: 0, v: 0 };

    // ---- controls: play, stops ----
    var playBtn = RL.el('button', { type: 'button', class: 'iconbtn' });
    var prevStop = iconBtn(ICON.left, 'Previous parameter');
    var nextStop = iconBtn(ICON.right, 'Next parameter');
    var seg = RL.el('div', { class: 'seg st-stops', role: 'group', 'aria-label': 'Parameter' });
    var stopBtns = stops.map(function (s, i) {
      var b = RL.el('button', { type: 'button', 'aria-pressed': 'false', 'aria-label': s.stitch + ': ' + s.variable },
        [RL.el('span', { class: 'st-full', text: SHORT[s.id] || s.variable }),
          RL.el('span', { class: 'st-tiny', text: TINY[s.id] || s.variable })]);
      b.addEventListener('click', function () { interact(); go(i, 0); });
      seg.appendChild(b);
      return b;
    });

    // ---- image pair, each a two-layer stack for crossfading ----
    function stack(label) {
      var imgs = [0, 1].map(function () {
        return RL.el('img', { alt: '', width: m.size, height: m.size, decoding: 'async', draggable: 'false' });
      });
      var fig = RL.el('figure', { class: 'st-fig' }, [RL.el('div', { class: 'st-img' }, imgs),
        RL.el('figcaption', { text: label })]);
      return { fig: fig, imgs: imgs, f: 0, timer: null };
    }
    var initL = stack('Procedural init');
    var finalL = stack('Generated fill (FLUX.1-dev + stitch LoRA)');

    // ---- value list ----
    var valHead = RL.el('div', { class: 'st-vhead' });
    var valUnit = RL.el('div', { class: 'st-vunit' });
    var prevVal = iconBtn(ICON.up, 'Previous value');
    var nextVal = iconBtn(ICON.down, 'Next value');
    var valList = RL.el('div', { class: 'st-vlist', role: 'group' });
    var valBtns = [];
    var values = RL.el('div', { class: 'st-values' }, [
      RL.el('div', { class: 'st-vtitle' }, [valHead, valUnit]), prevVal, valList, nextVal]);

    var caption = RL.el('p', { class: 'st-cap', 'aria-live': 'off' });

    root.innerHTML = '';
    root.setAttribute('tabindex', '0');
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', 'Parameter sweeps. Left and right arrow keys change the parameter; up and down change its value.');
    root.appendChild(RL.el('div', { class: 'st-controls' }, [playBtn, prevStop, seg, nextStop]));
    root.appendChild(RL.el('div', { class: 'st-body' }, [initL.fig, finalL.fig, values]));
    root.appendChild(caption);

    function urls(s) {
      var u = [];
      s.values.forEach(function (v) { u.push(BASE + v.init, BASE + v.final); });
      return u;
    }

    // Fade the new image in on top, then hide the old one underneath (no dip to the background).
    function swap(L, url, alt, instant) {
      var top = L.imgs[1 - L.f], under = L.imgs[L.f];
      clearTimeout(L.timer);
      top.style.transition = 'none';
      top.style.opacity = '0';
      top.src = url;
      top.alt = alt;
      under.alt = '';
      under.style.zIndex = '1';
      top.style.zIndex = '2';
      void top.offsetWidth;   // commit opacity 0 before the transition starts
      top.style.transition = instant ? 'none' : '';
      top.style.opacity = '1';
      L.f = 1 - L.f;
      L.timer = setTimeout(function () { under.style.opacity = '0'; }, instant ? 0 : FADE + 30);
    }

    function renderValues() {
      var s = stops[state.s];
      valHead.textContent = s.variable.charAt(0).toUpperCase() + s.variable.slice(1);
      valUnit.textContent = s.unit === 'px' ? 'px at 1024 px' : '';
      valList.setAttribute('aria-label', s.stitch + ' ' + s.variable);
      valList.innerHTML = '';
      valBtns = s.values.map(function (v, j) {
        var b = RL.el('button', { type: 'button', 'aria-pressed': 'false', text: valueText(s, v) });
        b.addEventListener('click', function () { interact(); go(state.s, j); });
        valList.appendChild(b);
        return b;
      });
      values.classList.toggle('wide', !s.unit);
    }

    function captionText(s, v) {
      var setting = s.id === 'colour' ? valueText(s, v) : s.variable + (s.unit ? ' ' : ': ') + valueText(s, v);
      return s.stitch + ' · ' + setting + ' · denoise ' + s.denoise + ' · seed fixed, ' + CHANGES[s.id];
    }

    var token = 0, builtFor = -1, first = true;
    function go(si, vi) {
      var n = stops.length;
      si = (si + n) % n;
      var s = stops[si];
      vi = (vi + s.values.length) % s.values.length;
      state.s = si; state.v = vi;
      if (builtFor !== si) {
        renderValues();
        builtFor = si;
        stopBtns.forEach(function (b, j) { b.setAttribute('aria-pressed', j === si ? 'true' : 'false'); });
        // Warm this stop, then the next one in the background.
        RL.loadAll(urls(s)).then(function () { RL.loadAll(urls(stops[(si + 1) % n])); });
      }
      valBtns.forEach(function (b, j) { b.setAttribute('aria-pressed', j === vi ? 'true' : 'false'); });
      var v = s.values[vi], my = ++token, what = s.stitch + ', ' + (s.id === 'colour' ? '' : s.variable + ' ') + valueText(s, v);
      caption.textContent = captionText(s, v);
      // Swap both images together once loaded, so the pair never mixes two settings.
      RL.loadAll([BASE + v.init, BASE + v.final]).then(function () {
        if (my !== token) return;
        var instant = first || RL.reducedMotion();
        first = false;
        swap(initL, BASE + v.init, 'Procedural init: ' + what + ', painted into the stitch regions of the castle', instant);
        swap(finalL, BASE + v.final, 'Generated fill from that init: ' + what + ', denoise ' + s.denoise, instant);
      });
    }

    // ---- playback ----
    var cycle = new RL.AutoCycle(root, STEP_MS, function () {
      var s = stops[state.s];
      if (state.v + 1 < s.values.length) go(state.s, state.v + 1);
      else go(state.s + 1, 0);
    });
    function setPlaying(p) {
      if (p) { cycle.stopped = false; cycle.resume(); } else cycle.stop();
      RL.setPlayButton(playBtn, p, 'the parameter sweeps');
      caption.setAttribute('aria-live', p ? 'off' : 'polite');
    }
    function interact() { if (!cycle.stopped) setPlaying(false); }

    playBtn.addEventListener('click', function () { setPlaying(cycle.stopped); });
    prevStop.addEventListener('click', function () { interact(); go(state.s - 1, 0); });
    nextStop.addEventListener('click', function () { interact(); go(state.s + 1, 0); });
    prevVal.addEventListener('click', function () { interact(); go(state.s, state.v - 1); });
    nextVal.addEventListener('click', function () { interact(); go(state.s, state.v + 1); });
    root.addEventListener('keydown', function (e) {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      var d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
      if (!d) return;
      e.preventDefault();
      interact();
      if (d[0]) go(state.s + d[0], 0); else go(state.s, state.v + d[1]);
    });

    setPlaying(!cycle.stopped);
    go(0, 0);
  }
})();
