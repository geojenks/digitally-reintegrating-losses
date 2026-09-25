/* "Reading the surface": public vs fine-tuned Marigold normals on held-out tiles (paper Fig. 2). */
(function () {
  'use strict';
  var root = document.getElementById('marigold');
  if (!root) return;
  var BASE = 'assets/marigold/';

  RL.whenNear(root, function () {
    RL.json(BASE + 'manifest.json').then(build).catch(function (e) {
      root.innerHTML = '<p class="widget-loading">Could not load the normal-estimation tiles.</p>';
      console.warn('marigold:', e);
    });
  });

  function build(tiles) {
    var PANELS = [
      { k: 'colour', cls: 'mg-c', label: 'Colour input' },
      { k: 'gt', cls: 'mg-g', label: 'Ground truth (PS + photogrammetry)' },
      { k: 'base', cls: 'mg-b', label: 'Public Marigold-Normals' },
      { k: 'ft', cls: 'mg-f', label: 'Our embroidery fine-tune' },
      { k: 'base_err', cls: 'mg-be', label: 'Angular error, public', mae: 'base_mae' },
      { k: 'ft_err', cls: 'mg-fe', label: 'Angular error, fine-tune', mae: 'ft_mae' }
    ];
    var main = RL.el('div', { class: 'mg-main' });
    var imgs = {}, maes = {};
    PANELS.forEach(function (p) {
      var img = RL.el('img', { alt: '', width: 384, height: 384 });
      var box = RL.el('div', { class: 'img' }, img);
      if (p.mae) { maes[p.k] = RL.el('span', { class: 'mae' }); box.appendChild(maes[p.k]); }
      main.appendChild(RL.el('figure', { class: 'mg-cell ' + p.cls }, [box, RL.el('figcaption', { text: p.label })]));
      imgs[p.k] = img;
    });
    main.appendChild(RL.el('div', { class: 'mg-cb' }, [
      RL.el('img', { src: BASE + 'colourbar.png', alt: 'Colour scale for angular error, 0 to 60 degrees', width: 512, height: 20 }),
      RL.el('div', { class: 'ticks' }, [RL.el('span', { text: '0°' }), RL.el('span', { text: 'angular error' }), RL.el('span', { text: '60°' })])
    ]));
    var bigBase = RL.el('span'), bigFt = RL.el('span', { class: 'ft' }), tileLine = RL.el('div', { class: 'tile' });
    main.appendChild(RL.el('div', { class: 'mg-stats' }, [
      RL.el('div', { text: 'Mean angular error on this tile' }),
      RL.el('div', { class: 'big' }, [bigBase, RL.el('span', { class: 'arrow', text: '→' }), bigFt]),
      RL.el('div', { class: 'muted', style: 'font-size:.85em', text: 'public Marigold-Normals → our fine-tune' }),
      tileLine
    ]));

    var thumbs = RL.el('div', { class: 'mg-thumbs', role: 'group', 'aria-label': 'Held-out tiles' });
    var tbtns = tiles.map(function (t, i) {
      var b = RL.el('button', { type: 'button', 'aria-pressed': 'false',
        'aria-label': 'Tile ' + t.tile + (t.paper ? ' (in the paper’s figure)' : ''), title: t.tile },
        [RL.el('img', { src: BASE + t.files.colour, alt: '', loading: 'lazy', width: 384, height: 384 }),
          t.paper ? RL.el('span', { class: 'dot', 'aria-hidden': 'true' }) : null]);
      b.addEventListener('click', function () { cycle.stop(); show(i); });
      thumbs.appendChild(b);
      return b;
    });

    root.innerHTML = '';
    root.appendChild(main);
    root.appendChild(thumbs);
    root.appendChild(RL.el('p', { class: 'cap left' }, [
      'Pick a tile. The three marked ',
      RL.el('span', { style: 'display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent)' }),
      ' are the ones shown in the paper’s Figure 2.'
    ]));

    var current = -1, token = 0;
    function urls(t) { return PANELS.map(function (p) { return BASE + t.files[p.k]; }); }
    function show(i) {
      var t = tiles[i], my = ++token;
      tbtns.forEach(function (b, j) { b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
      // Swap all six images together once loaded, so the row never mixes two tiles.
      RL.loadAll(urls(t)).then(function () {
        if (my !== token) return;
        current = i;
        PANELS.forEach(function (p) {
          imgs[p.k].src = BASE + t.files[p.k];
          imgs[p.k].alt = p.label + ' for held-out tile ' + t.tile +
            (p.mae ? ' (mean angular error ' + RL.fmt(t[p.mae], 1) + '°)' : '');
          if (p.mae) maes[p.k].textContent = RL.fmt(t[p.mae], 1) + '°';
        });
        bigBase.textContent = RL.fmt(t.base_mae, 1) + '°';
        bigFt.textContent = RL.fmt(t.ft_mae, 1) + '°';
        tileLine.innerHTML = '';
        tileLine.appendChild(document.createTextNode('Held-out tile ' + t.tile));
        if (t.paper) tileLine.appendChild(RL.el('span', { class: 'paper-tag', text: 'in the paper’s figure' }));
        RL.loadAll(urls(tiles[(i + 1) % tiles.length]));   // warm the next tile
      });
    }

    var cycle = new RL.AutoCycle(root, 4000, function () { show((current + 1) % tiles.length); });
    root.addEventListener('keydown', function () { cycle.stop(); });
    show(0);
  }
})();
