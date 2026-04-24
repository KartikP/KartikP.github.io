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
}

export function learningFiringRate(neuronIndex) {
    const s = learningState;
    if (!s.burstActive) return 0;
    if (!s.burstNeuronMask[neuronIndex]) return 0;

    // Gaussian envelope peaking at burst midpoint.
    const elapsed = s.burstDuration - s.burstFramesLeft;
    const center = s.burstDuration / 2;
    const sigma = Math.max(s.burstDuration / 4, 1);
    const envelope = Math.exp(
        -((elapsed - center) * (elapsed - center)) / (2 * sigma * sigma)
    );
    return s.burstBaseProb * envelope;
}

export function learningIsBurstActive() {
    return learningState.burstActive;
}
