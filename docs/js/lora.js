/* "Learning a stitch": LoRA training samples across four models (paper Fig. 3). */
(function () {
  'use strict';
  var root = document.getElementById('lora');
  if (!root) return;
  var BASE = 'assets/lora_training/';
  var FRAME_MS = 700, END_HOLD_MS = 2500, MAX_STEP = 7000, STEP = 500;
  var STITCH_LABEL = { french_knot: 'French knot', satin: 'Satin', silk_purl: 'Silk purl' };

  RL.whenNear(root, function () {
    RL.json(BASE + 'manifest.json').then(build).catch(function (e) {
      root.innerHTML = '<p class="widget-loading">Could not load the training frames.</p>';
      console.warn('lora:', e);
    });
  });

  function build(m) {
    var models = {};
    m.models.forEach(function (mo) { models[mo.key] = mo; });
    var still = RL.reducedMotion();
    var state = { flux: 'flux1_v1', prompt: 'paper', step: still ? MAX_STEP : 0, playing: !still };

    // ---- controls ----
    var playBtn = RL.el('button', { type: 'button', class: 'iconbtn' });
    var slider = RL.el('input', { type: 'range', min: 0, max: MAX_STEP, step: STEP, value: 0,
      'aria-label': 'Training step' });
    var stepOut = RL.el('output', { class: 'lora-step', 'aria-live': 'off' });
    var promptSel = RL.el('select', { 'aria-label': 'Sampling prompt' }, [
      RL.el('option', { value: 'paper', text: 'as in the paper' }),
      RL.el('option', { value: '0', text: 'trigger word only' }),
      RL.el('option', { value: '1', text: 'trigger + “close-up macro photograph, raking light”' })
    ]);
    var controls = RL.el('div', { class: 'lora-controls' }, [
      RL.el('div', { class: 'lora-scrub' }, [playBtn, slider, stepOut]),
      RL.el('label', { class: 'ctl-label' }, ['Prompt', promptSel])
    ]);

    // ---- grid ----
    var grid = RL.el('div', { class: 'lora-grid' });
    var cols = ['sdxl', 'flux', 'qwen', 'zimage'];
    var colLabel = { sdxl: 'SDXL', flux: 'FLUX.1-dev', qwen: 'Qwen-Image', zimage: 'Z-Image' };
    grid.appendChild(RL.el('div'));
    grid.appendChild(RL.el('div', { class: 'colhead gt', text: 'Ground truth' }));
    cols.forEach(function (c) { grid.appendChild(RL.el('div', { class: 'colhead', text: colLabel[c] })); });
    var cells = {};
    m.stitches.forEach(function (st) {
      grid.appendChild(RL.el('div', { class: 'rowhead', text: STITCH_LABEL[st] }));
      grid.appendChild(RL.el('figure', { class: 'lora-cell gt' },
        RL.el('img', { src: BASE + m.gt[st], alt: 'Ground-truth ' + STITCH_LABEL[st] + ' crop from the corpus',
          width: 320, height: 320, loading: 'lazy' })));
      cols.forEach(function (c) {
        var img = RL.el('img', { alt: '', width: 320, height: 320 });
        var tag = RL.el('span', { class: 'tag', text: 'not sampled' });
        var fig = RL.el('figure', { class: 'lora-cell' }, [img, tag]);
        grid.appendChild(fig);
        cells[st + '|' + c] = { fig: fig, img: img, tag: tag, st: st, col: c, shown: '' };
      });
    });
    var promptNote = RL.el('p', { class: 'lora-prompt' });

    root.innerHTML = '';
    root.appendChild(controls);
    root.appendChild(grid);
    root.appendChild(promptNote);

    function modelKey(c) { return c === 'flux' ? state.flux : c; }
    function promptIdx(key) { return state.prompt === 'paper' ? m.figure3_prompt_index[key] : +state.prompt; }

    // Latest sampled frame at or before `step` for one cell.
    function frameFor(cell, step) {
      var key = modelKey(cell.col), run = models[key].runs[cell.st], p = String(promptIdx(key));
      var k = -1;
      for (var i = 0; i < run.steps.length; i++) if (run.steps[i] <= step) k = i;
      return { file: run.files[p][k], at: run.steps[k], exact: run.steps[k] === step,
        prompt: run.prompts[p], label: models[key].label };
    }

    function configUrls() {
      var urls = [];
      Object.keys(cells).forEach(function (id) {
        var c = cells[id], key = modelKey(c.col), run = models[key].runs[c.st];
        run.files[String(promptIdx(key))].forEach(function (f) { urls.push(BASE + f); });
      });
      return urls;
    }

    function render() {
      var s = state.step;
      slider.value = s;
      stepOut.textContent = s === 0 ? 'step 0 (base model)' : 'step ' + s.toLocaleString('en-GB') + ' / 7,000';
      Object.keys(cells).forEach(function (id) {
        var c = cells[id], f = frameFor(c, s);
        var src = BASE + f.file;
        if (c.shown !== src) { c.img.src = src; c.shown = src; }
        c.fig.classList.toggle('stale', !f.exact);
        var desc = f.label + ', ' + STITCH_LABEL[c.st] + ', prompt “' + f.prompt + '”, ' +
          (f.exact ? 'training step ' + f.at : 'no sample at step ' + s + '; showing step ' + f.at);
        c.img.alt = desc;
        c.fig.title = desc;
      });
    }

    function renderPromptNote() {
      var trig = '<code>embfnchknt</code>, <code>embstn</code>, <code>embslkprl</code>';
      var extra = '<code>close-up macro photograph, raking light</code>';
      var txt;
      if (state.prompt === 'paper') {
        txt = 'Prompts as in the paper’s figure: SDXL samples use the trigger word + ' + extra +
          '; the other models use the trigger word alone (' + trig + ').';
      } else if (state.prompt === '0') {
        txt = 'All models prompted with the trigger word alone (' + trig + ').';
      } else {
        txt = 'All models prompted with the trigger word + ' + extra + '.';
      }
      promptNote.innerHTML = txt + ' Samples every 500 steps, fixed seed per prompt; step 0 is the untrained base model. ' +
        'Dimmed cells tagged “not sampled” hold the latest earlier sample: Qwen-Image satin was not sampled at 4,500 to 6,500,' +
        'and Qwen-Image silk purl only at 0 and 7,000.';
    }

    // ---- playback ----
    var timer = null, onScreen = false, loadedFor = '';
    function schedule() {
      clearTimeout(timer); timer = null;
      if (!state.playing || !onScreen || loadedFor !== configId()) return;
      timer = setTimeout(function () {
        state.step = state.step >= MAX_STEP ? 0 : state.step + STEP;
        render();
        schedule();
      }, state.step >= MAX_STEP ? END_HOLD_MS : (state.step === 0 ? FRAME_MS * 1.8 : FRAME_MS));
    }
    function configId() { return state.flux + '|' + state.prompt; }
    function setPlaying(p) {
      state.playing = p;
      RL.setPlayButton(playBtn, p, 'training steps');
      schedule();
    }
    function reconfigure() {
      var id = configId();
      render();
      renderPromptNote();
      RL.loadAll(configUrls()).then(function () {
        if (configId() !== id) return;
        loadedFor = id;
        schedule();
      });
    }

    playBtn.addEventListener('click', function () {
      if (!state.playing && state.step >= MAX_STEP) { state.step = 0; render(); }
      setPlaying(!state.playing);
    });
    slider.addEventListener('input', function () {
      setPlaying(false);
      state.step = +slider.value;
      render();
    });
    promptSel.addEventListener('change', function () { state.prompt = promptSel.value; reconfigure(); });

    RL.onVisibility(grid, function (v) { onScreen = v; schedule(); });
    RL.setPlayButton(playBtn, state.playing, 'training steps');
    reconfigure();
  }
})();
