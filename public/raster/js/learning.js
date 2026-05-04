// learning.js — progressive network-burst behaviour as the simulation
// "learns" from repeated RSBs triggered by the user.
//
// Core idea (a Hebbian gesture, not a real Hebbian rule):
//   - Every RSB that completes nudges a global `strength` toward 1.
//   - `strength` slowly decays over time so the effect is session-local.
//   - While `strength > 0`, the idle state occasionally spawns a
//     spontaneous network burst. Frequency, neuron participation, and
//     burst amplitude all scale with `strength`.
//
// Public API:
//   learningState                          — inspectable state object
//   learningTick(rowCount, rsbPhase)       — call once per frame before neurons
//   learningFiringRate(neuronIndex)        — per-neuron extra firing prob.
//   learningIsBurstActive()                — true during a spontaneous burst

export const learningState = {
    strength: 0,
    rsbCount: 0,

    burstActive: false,
    burstFramesLeft: 0,
    burstDuration: 0,
    burstNeuronMask: [],
    burstBaseProb: 0,

    _lastRsbPhase: 'idle',
};

// Optional registration so a burst-start event can include per-target
// intensities (used by the landing-page nav glitch). Without registration
// the burst event still fires but the intensities array is empty.
//
// Register-time precomputation: the angular-tuning weights between every
// (neuron, nav target) pair are fixed for the session — preferred angles
// don't change after init. We pre-compute them once into Float32Arrays so
// per-burst intensity is a pure conditional-add over the mask, no Math.exp,
// no Math.abs, on the burst-start frame.
let _navAngles = null;
let _navWeights = null;     // Array<Float32Array(neuronCount)>, one per target
let _navTotalWeights = null; // Float32Array(navCount)
const PER_TARGET_TUNING = 0.45; // Gaussian σ over angular distance

export function learningRegisterTargets(neuronAngles, navAngles) {
    _navAngles = navAngles;
    const sigma2 = 2 * PER_TARGET_TUNING * PER_TARGET_TUNING;
    _navWeights = new Array(navAngles.length);
    _navTotalWeights = new Float32Array(navAngles.length);
    for (let t = 0; t < navAngles.length; t++) {
        const phi = navAngles[t];
        const arr = new Float32Array(neuronAngles.length);
        let total = 0;
        for (let i = 0; i < neuronAngles.length; i++) {
            let d = Math.abs(neuronAngles[i] - phi);
            if (d > Math.PI) d = Math.PI * 2 - d;
            const w = Math.exp(-(d * d) / sigma2);
            arr[i] = w;
            total += w;
        }
        _navWeights[t] = arr;
        _navTotalWeights[t] = total;
    }
}

function computePerTargetIntensities(mask) {
    if (!_navWeights) return [];
    const out = new Array(_navWeights.length);
    for (let t = 0; t < _navWeights.length; t++) {
        const w = _navWeights[t];
        const total = _navTotalWeights[t];
        let firing = 0;
        for (let i = 0; i < w.length; i++) {
            if (mask[i]) firing += w[i];
        }
        out[t] = total > 0 ? firing / total : 0;
    }
    return out;
}

// Tuning constants.
const STRENGTH_GAIN_PER_RSB = 0.15;   // asymptotic: s += (1 - s) * GAIN
const STRENGTH_DECAY = 0.99998;       // per-frame; half-life ~34,000 frames (~10 min at 60fps)
const BURST_START_GAIN = 0.005;       // per-frame prob scalar
const BURST_START_EXP = 0.7;          // sub-linear ramp so early strength still fires
const RECRUIT_BASE = 0.05;
const RECRUIT_GAIN = 0.55;
const BURST_PROB_BASE = 0.02;
const BURST_PROB_GAIN = 0.23;
const BURST_DUR_BASE = 15;
const BURST_DUR_GAIN = 25;
const ACTIVATION_FLOOR = 0.05;        // below this, don't attempt bursts

export function learningTick(rowCount, rsbPhase) {
    const s = learningState;

    // RSB-completion detection: any non-idle phase -> idle.
    if (s._lastRsbPhase !== 'idle' && rsbPhase === 'idle') {
        s.rsbCount++;
        s.strength += (1 - s.strength) * STRENGTH_GAIN_PER_RSB;
    }
    s._lastRsbPhase = rsbPhase;

    // Slow decay so the effect is session-local rather than permanent.
    s.strength *= STRENGTH_DECAY;

    // Don't start spontaneous bursts during an RSB — they'd compete visually.
    const canStart = !s.burstActive
        && rsbPhase === 'idle'
        && s.strength > ACTIVATION_FLOOR;

    if (canStart) {
        const startProb = Math.pow(s.strength, BURST_START_EXP) * BURST_START_GAIN;
        if (Math.random() < startProb) {
            startSpontaneousBurst(rowCount);
        }
    }

    if (s.burstActive) {
        s.burstFramesLeft--;
        if (s.burstFramesLeft <= 0) {
            s.burstActive = false;
            s.burstNeuronMask.length = 0;
        }
    }
}

function startSpontaneousBurst(rowCount) {
    const s = learningState;
    s.burstDuration = Math.round(BURST_DUR_BASE + s.strength * BURST_DUR_GAIN);
    s.burstFramesLeft = s.burstDuration;
    s.burstActive = true;
    s.burstBaseProb = BURST_PROB_BASE + s.strength * BURST_PROB_GAIN;

    const recruitFrac = RECRUIT_BASE + s.strength * RECRUIT_GAIN;

    // Fresh mask — which subset of neurons participates in THIS burst.
    const mask = new Array(rowCount);
    for (let i = 0; i < rowCount; i++) {
        mask[i] = Math.random() < recruitFrac ? 1 : 0;
    }
    s.burstNeuronMask = mask;

    // Notify any UI that wants to react to spontaneous bursts (the landing
    // page nav titles use this to apply a brief chromatic-split glitch
    // whose intensity tracks the fraction of each target's tuned neurons
    // that fired in this burst). Deferred off the rAF frame so the
    // intensity sum and listener fan-out don't stutter the frame the
    // burst started on.
    if (typeof window !== "undefined" && typeof CustomEvent !== "undefined") {
        const burstSnapshot = {
            strength: s.strength,
            duration: s.burstDuration,
            recruitFrac: recruitFrac,
        };
        setTimeout(function () {
            const intensities = computePerTargetIntensities(mask);
            window.dispatchEvent(new CustomEvent("raster:burst", {
                detail: Object.assign(burstSnapshot, { intensities: intensities }),
            }));
        }, 0);
    }
}

// Time-only envelope shared across every neuron in the current burst.
// Hoisted out of the per-neuron firing rate so the renderer can compute
// it once per frame instead of once per (neuron × frame). Saves ~999
// Math.exp calls per burst frame at ROW_COUNT = 1000.
export function learningBurstEnvelope() {
    const s = learningState;
    if (!s.burstActive) return 0;
    const elapsed = s.burstDuration - s.burstFramesLeft;
    const center = s.burstDuration / 2;
    const sigma = Math.max(s.burstDuration / 4, 1);
    return Math.exp(
        -((elapsed - center) * (elapsed - center)) / (2 * sigma * sigma)
    );
}

// Returns the per-neuron learned-burst rate. Kept for external callers
// but the renderer now uses the hoisted envelope path.
export function learningFiringRate(neuronIndex) {
    const s = learningState;
    if (!s.burstActive) return 0;
    if (!s.burstNeuronMask[neuronIndex]) return 0;
    return s.burstBaseProb * learningBurstEnvelope();
}

export function learningIsBurstActive() {
    return learningState.burstActive;
}
