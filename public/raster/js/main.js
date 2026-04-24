import { ROW_COUNT } from './config.js';
import { schemes } from './schemes.js';
import { createNeurons } from './neurons.js';
import { bindInputEvents } from './input.js';
import { bindRsbEvents } from './rsb.js';
import { createRenderer } from './renderer.js';

// Module-scoped renderer so we can tear it down on View Transition navigation.
let currentRenderer = null;
let schemeListener = null;

function teardown() {
    if (currentRenderer) {
        currentRenderer.stop();
        currentRenderer = null;
    }
    if (schemeListener) {
        const el = document.getElementById('colorScheme');
        if (el) el.removeEventListener('change', schemeListener);
        schemeListener = null;
    }
}

function init() {
    const rCanvas = document.getElementById('rasterCanvas');
    if (!rCanvas) return; // not on a page that has the raster

    // If a prior instance exists (user navigated back to landing), tear it down.
    teardown();

    const dom = {
        rCanvas,
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

    let currentScheme = schemes.greyscale;
    const neurons = createNeurons(ROW_COUNT);

    const schemeSelect = document.getElementById('colorScheme');
    if (schemeSelect) {
        schemeListener = () => {
            currentScheme = schemes[schemeSelect.value];
        };
        schemeSelect.addEventListener('change', schemeListener);
    }

    // bindInput/Rsb are idempotent — harmless if called again after navigation.
    bindInputEvents();
    bindRsbEvents(ROW_COUNT);

    const renderer = createRenderer(dom, neurons, () => currentScheme);
    renderer.tick();
    currentRenderer = renderer;

    // Safety net: fire resize a few times after init so whenever the flex
    // layout lands (could be mid-View-Transition, could be after), the
    // canvas buffers sync to the correct CSS dimensions. resize() is
    // idempotent, so the redundant fires are no-ops once dimensions settle.
    [0, 100, 300, 600].forEach((ms) => {
        setTimeout(() => {
            if (currentRenderer === renderer) renderer.resize();
        }, ms);
    });
}

// Run on first script execution AND on every Astro View Transition page load.
init();
document.addEventListener('astro:page-load', init);

// Always tear down before the DOM is swapped — init() on the next page's
// astro:page-load will recreate the renderer if the new page has the canvas.
document.addEventListener('astro:before-swap', teardown);
