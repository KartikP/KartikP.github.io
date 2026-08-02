// hover.js — transient hover-evoked selective firing.
//
// Each interactive element on the landing has a "preferred angle" in
// the same 2π space as the neurons' direction-tuning. Hovering an
// element produces a brief, sharp flash of activity in the small
// subpopulation whose preferred angle matches — modelled like a
// stimulus-onset response with fast rise and exponential decay.
//
// Behavior summary:
//   - Mouseover a button → strong but very sparse selective firing
//   - Decays to ~0 over ~300 ms regardless of whether you stay hovering
//   - Re-entering the same button (after leaving) re-triggers the flash
//   - Moving within the same button does NOT re-trigger
//   - Moving directly between buttons triggers a fresh flash for the new one

export const hoverState = {
    active: false,
    angle: 0,
    buttonId: null,
};

let hoverStartTimeMs = 0;
let cachedNeurons = null;
let cachedAngle = NaN;
let cachedTuning = null;

// Sparseness controls — lower = sparser & sharper.
const HOVER_TUNING_WIDTH = 0.15;     // gaussian width on the angular tuning curve
const HOVER_PEAK_INTENSITY = 0.7;    // firing-rate scalar at t=0
const HOVER_DECAY_TAU_MS = 130;      // exponential time constant; ~5τ ≈ 650ms

export function setHoverActive(buttonId, angle) {
    // Don't re-trigger if we're already on this button (prevents
    // mouseover-on-children events from re-flashing every nudge).
    if (hoverState.active && hoverState.buttonId === buttonId) return;
    hoverState.active = true;
    hoverState.buttonId = buttonId;
    hoverState.angle = angle;
    hoverStartTimeMs = performance.now();
}

export function setHoverInactive() {
    hoverState.active = false;
    hoverState.buttonId = null;
}

// Prepare the hover response once per animation frame. The tuning curve only
// depends on the active target and each neuron's fixed preferred angle, so it
// is rebuilt only when either of those inputs changes. Previously both the
// decay and tuning exponentials were evaluated for every neuron, every frame.
export function prepareHoverFrame(neurons) {
    if (!hoverState.active) return 0;

    const elapsed = performance.now() - hoverStartTimeMs;
    const decay = Math.exp(-elapsed / HOVER_DECAY_TAU_MS);
    if (decay < 0.01) return 0;

    if (cachedNeurons !== neurons || cachedAngle !== hoverState.angle) {
        const sigma2 = 2 * HOVER_TUNING_WIDTH * HOVER_TUNING_WIDTH;
        const tuning = new Float64Array(neurons.length);
        for (let i = 0; i < neurons.length; i++) {
            let dist = Math.abs(hoverState.angle - neurons[i].preferredAngle);
            if (dist > Math.PI) dist = Math.PI * 2 - dist;
            tuning[i] = Math.exp(-(dist * dist) / sigma2);
        }
        cachedNeurons = neurons;
        cachedAngle = hoverState.angle;
        cachedTuning = tuning;
    }

    return HOVER_PEAK_INTENSITY * decay;
}

export function hoverFiringRate(neuronIndex, hoverScalar) {
    if (!cachedTuning || hoverScalar === 0) return 0;
    return cachedTuning[neuronIndex] * hoverScalar;
}

export function hoverIsActive() {
    return hoverState.active;
}
