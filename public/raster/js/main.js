import { ROW_COUNT } from './config.js';
import { schemes } from './schemes.js';
import { createNeurons } from './neurons.js';
import { bindInputEvents } from './input.js';
import { bindRsbEvents } from './rsb.js';
import { createRenderer } from './renderer.js';

// DOM elements
const dom = {
    rCanvas: document.getElementById('rasterCanvas'),
    lCanvas: document.getElementById('lfpCanvas'),
    mCanvas: document.getElementById('minimapCanvas'),
    statStatus: document.getElementById('stat-status'),
    statPop: document.getElementById('stat-pop'),
    statDir: document.getElementById('stat-dir'),
    statCount: document.getElementById('stat-count'),
    pulseDot: document.getElementById('pulse'),
};

dom.rCtx = dom.rCanvas.getContext('2d', { alpha: false });
dom.lCtx = dom.lCanvas.getContext('2d', { alpha: false });
dom.mCtx = dom.mCanvas.getContext('2d');

// State
let currentScheme = schemes.greyscale;
const neurons = createNeurons(ROW_COUNT);

// Scheme dropdown
const schemeSelect = document.getElementById('colorScheme');
schemeSelect.addEventListener('change', () => {
    currentScheme = schemes[schemeSelect.value];
});

// Bind input events
bindInputEvents();
bindRsbEvents(ROW_COUNT);

// Create renderer and start
const renderer = createRenderer(dom, neurons, () => currentScheme);
renderer.tick();
