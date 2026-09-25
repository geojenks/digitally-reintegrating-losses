/* Shared helpers for the project-page widgets. No dependencies. */
(function () {
  'use strict';
  var RL = (window.RL = {});

  var mq = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  RL.reducedMotion = function () { return !!(mq && mq.matches); };

  /* el('div', {class: 'x', onclick: fn}, [children]) */
  RL.el = function (tag, attrs, kids) {
    var n = document.createElement(tag);
    setAttrs(n, attrs);
    append(n, kids);
    return n;
  };
  var SVGNS = 'http://www.w3.org/2000/svg';
  RL.svg = function (tag, attrs, kids) {
    var n = document.createElementNS(SVGNS, tag);
    setAttrs(n, attrs);
    append(n, kids);
    return n;
  };
  function setAttrs(n, attrs) {
    if (!attrs) return;
    for (var k in attrs) {
      if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
      var v = attrs[k];
      if (v === null || v === undefined || v === false) continue;
      if (k.slice(0, 2) === 'on' && typeof v === 'function') n.addEventListener(k.slice(2), v);
      else if (k === 'text') n.textContent = v;
      else if (k === 'html') n.innerHTML = v;
      else n.setAttribute(k, v === true ? '' : v);
    }
  }
  function append(n, kids) {
    if (kids === null || kids === undefined) return;
    if (!Array.isArray(kids)) kids = [kids];
    kids.forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
  }

  RL.json = function (url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
      return r.json();
    });
  };

  /* Run cb once, when el comes within `margin` of the viewport. */
  RL.whenNear = function (el, cb, margin) {
    if (!('IntersectionObserver' in window)) { cb(); return; }
    var io = new IntersectionObserver(function (entries) {
      if (entries.some(function (e) { return e.isIntersecting; })) { io.disconnect(); cb(); }
    }, { rootMargin: margin || '400px 0px' });
    io.observe(el);
  };

  /* Call cb(true/false) as el enters/leaves the viewport. */
  RL.onVisibility = function (el, cb) {
    if (!('IntersectionObserver' in window)) { cb(true); return; }
    new IntersectionObserver(function (entries) {
      cb(entries[entries.length - 1].isIntersecting);
    }, { threshold: 0.05 }).observe(el);
  };

  /* Resolve when the image at url has loaded (or failed); caches by url. */
  var cache = {};
  RL.load = function (url) {
    if (!cache[url]) {
      cache[url] = new Promise(function (res) {
        var i = new Image();
        i.decoding = 'async';
        i.onload = function () { res(i); };
        i.onerror = function () { res(null); };
        i.src = url;
      });
    }
    return cache[url];
  };
  RL.loadAll = function (urls) { return Promise.all(urls.map(RL.load)); };

  var ICON_PLAY = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5v11l9-5.5z"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M2.5 1.5h3.2v11H2.5zM8.3 1.5h3.2v11H8.3z"/></svg>';
  /* Set a play/pause toggle button's icon and label. */
  RL.setPlayButton = function (btn, playing, what) {
    btn.innerHTML = (playing ? ICON_PAUSE : ICON_PLAY) + '<span class="visually-hidden"></span>';
    btn.lastChild.textContent = (playing ? 'Pause ' : 'Play ') + (what || '');
    btn.setAttribute('aria-pressed', playing ? 'true' : 'false');
    btn.title = playing ? 'Pause' : 'Play';
  };

  /*
   * Auto-advance helper: calls step() every `ms` while the widget is on screen and the tab
   * is visible, until stop() is called (on first user interaction). pause()/resume() are for
   * transient holds such as hovering.
   */
  RL.AutoCycle = function (el, ms, step) {
    var self = this, timer = null, onScreen = false, held = false;
    this.stopped = RL.reducedMotion();
    function tick() { timer = null; if (self.running()) { step(); arm(); } }
    function arm() { if (!timer && self.running()) timer = setTimeout(tick, ms); }
    function disarm() { if (timer) { clearTimeout(timer); timer = null; } }
    this.running = function () { return !self.stopped && !held && onScreen && !document.hidden; };
    this.stop = function () { self.stopped = true; disarm(); };
    this.pause = function () { held = true; disarm(); };
    this.resume = function () { held = false; arm(); };
    RL.onVisibility(el, function (v) { onScreen = v; if (v) arm(); else disarm(); });
    document.addEventListener('visibilitychange', function () { if (document.hidden) disarm(); else arm(); });
  };

  RL.fmt = function (x, d) { return Number(x).toFixed(d === undefined ? 2 : d); };
})();
