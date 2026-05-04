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

export function hoverFiringRate(neuron) {
    if (!hoverState.active) return 0;

    // Decay envelope — flash fades to ~0 over ~5τ.
    const elapsed = performance.now() - hoverStartTimeMs;
    const decay = Math.exp(-elapsed / HOVER_DECAY_TAU_MS);
    if (decay < 0.01) return 0; // gate: don't bother computing tuning once faded

    // Angular selectivity (narrow gaussian).
    let dist = Math.abs(hoverState.angle - neuron.preferredAngle);
    if (dist > Math.PI) dist = Math.PI * 2 - dist;
    const tuning = Math.exp(
        -(dist * dist) / (2 * HOVER_TUNING_WIDTH * HOVER_TUNING_WIDTH)
    );

    return HOVER_PEAK_INTENSITY * tuning * decay;
}

export function hoverIsActive() {
    return hoverState.active;
}
