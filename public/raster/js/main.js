import { ROW_COUNT } from './config.js';
import { schemes } from './schemes.js';
import { createNeurons } from './neurons.js';
import { bindInputEvents } from './input.js';
import { bindRsbEvents } from './rsb.js';
import { setHoverActive, setHoverInactive } from './hover.js';
import { learningRegisterTargets } from './learning.js';
import { createRenderer } from './renderer.js';

// Each interactive element has a "preferred angle" in the same 2π space
// as the neurons' direction-tuning. Hovering the element drives every
// neuron whose preferred angle is close — same direction-selectivity the
// cursor movement uses, but driven by a fixed angle and through a sparser
// tuning curve (see HOVER_TUNING_WIDTH in hover.js) so only well-matched
// neurons respond.
//
// Semantic similarity → angular proximity → neuron-population overlap:
//
//   About (0.40) ── WIT (0.55)              "self / context" cluster
//                                           (0.15 rad apart → ~75% overlap)
//
//   Posts (3.30) ── Projects (3.40) ── Publications (3.55)
//                                           "work output" cluster
//   • Posts ↔ Projects (0.10 rad)            ~88% overlap
//   • Projects ↔ Publications (0.15 rad)     ~75% overlap
//   • Posts ↔ Publications (0.25 rad)        ~46% overlap (more distinct)
//
// The two clusters are ~π apart → essentially disjoint populations.
const NAV_TARGETS = [
    { id: 'about',       angle: 0.40 },
    { id: 'posts',       angle: 3.30 },
    { id: 'projects',    angle: 3.40 },
    { id: 'publications', angle: 3.55 },
];
const WIT_TARGET = { id: 'wit', angle: 0.55 };
// Intensity, decay tau, and tuning width all live in hover.js as constants.

function bindHoverEvents() {
    if (window.__navHoverBound) return; // idempotent — survives View Transitions
    window.__navHoverBound = true;

    function targetFor(el) {
        if (!el || !el.closest) return null;
        const wit = el.closest('.wit-button');
        if (wit) return { el: wit, target: WIT_TARGET };
        const link = el.closest('.landing-nav a');
        if (link) {
            const links = document.querySelectorAll('.landing-nav a');
            for (let i = 0; i < links.length; i++) {
                if (links[i] === link) {
                    return { el: link, target: NAV_TARGETS[i] ?? NAV_TARGETS[0] };
                }
            }
        }
        return null;
    }

    document.addEventListener('mouseover', (e) => {
        const t = targetFor(e.target);
        if (t) setHoverActive(t.target.id, t.target.angle);
    });

    document.addEventListener('mouseout', (e) => {
        const t = targetFor(e.target);
        if (!t) return;
        // Only deactivate if we're actually leaving the hovered element,
        // not just crossing into one of its children.
        const related = e.relatedTarget;
        if (!related || !t.el.contains(related)) {
            setHoverInactive();
        }
    });
}

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
        glowCanvas: document.getElementById('lfpGlowCanvas'),
    };
    dom.rCtx = dom.rCanvas.getContext('2d', { alpha: false });
    dom.lCtx = dom.lCanvas.getContext('2d', { alpha: false });
    dom.gCtx = dom.glowCanvas.getContext('2d');

    let currentScheme = schemes.greyscale;
    const neurons = createNeurons(ROW_COUNT);

    // Register per-target angles so spontaneous-burst events can carry
    // per-link intensities (landing nav glitch reads this).
    learningRegisterTargets(
        neurons.map((n) => n.preferredAngle),
        NAV_TARGETS.map((t) => t.angle),
    );

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
    bindHoverEvents();

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
