export const rsbState = {
    holding: false,
    phase: 'idle',       // 'idle' | 'holding' | 'decaying' | 'gap' | 'miniburst'
    framesLeft: 0,
    miniburstsLeft: 0,
    miniburstsTotal: 0,
    currentProb: 0,
    peakProb: 0,
    holdFrames: 0,
    neuronMask: [],
    miniProb: 0,
    miniDuration: 0,
};

export function rsbStart(rowCount) {
    const s = rsbState;
    if (s.phase !== 'idle') return;
    s.holding = true;
    s.phase = 'holding';
    s.currentProb = 0.15;
    s.peakProb = 0;
    s.holdFrames = 0;
    s.neuronMask = new Array(rowCount).fill(0);
}

export function rsbRelease() {
    const s = rsbState;
    if (!s.holding) return;
    s.holding = false;
    if (s.phase === 'holding') {
        s.phase = 'decaying';
        s.peakProb = s.currentProb;
        const scale = Math.min(s.holdFrames / 180, 1);
        s.miniburstsTotal = Math.round(2 + scale * 8);
        s.miniburstsLeft = s.miniburstsTotal;
        s.miniProb = 0.06 + scale * 0.15;
    }
}

export function rsbTick(rowCount) {
    const s = rsbState;

    if (s.phase === 'holding') {
        s.holdFrames++;
        s.currentProb += (0.7 - s.currentProb) * 0.18;
        const recruited = s.neuronMask.reduce((sum, v) => sum + v, 0);
        const recruitRate = 0.08 * (1 - recruited / rowCount) + 0.005;
        for (let j = 0; j < rowCount; j++) {
            if (s.neuronMask[j] < 1 && Math.random() < recruitRate) {
                s.neuronMask[j] = 1;
            }
        }
        return s.currentProb;
    }

    if (s.phase === 'decaying') {
        s.currentProb *= 0.94;
        if (s.currentProb < 0.01) {
            s.phase = 'gap';
            s.framesLeft = Math.round(8 + Math.random() * 10);
        }
        return s.currentProb;
    }

    if (s.phase === 'gap') {
        s.framesLeft--;
        if (s.framesLeft <= 0) {
            if (s.miniburstsLeft > 0) {
                s.phase = 'miniburst';
                s.miniDuration = Math.round(14 + Math.random() * 12);
                s.framesLeft = s.miniDuration;
                s.miniburstsLeft--;
            } else {
                s.phase = 'idle';
            }
        }
        return 0;
    }

    if (s.phase === 'miniburst') {
        const elapsed = s.miniDuration - s.framesLeft;
        const center = s.miniDuration / 2;
        const sigma = s.miniDuration / 4.5;
        const envelope = Math.exp(-Math.pow(elapsed - center, 2) / (2 * sigma * sigma));
        const burstIndex = s.miniburstsTotal - s.miniburstsLeft - 1;
        const ampDecay = Math.exp(-3 * burstIndex / Math.max(s.miniburstsTotal - 1, 1));
        s.framesLeft--;
        if (s.framesLeft <= 0) {
            s.phase = 'gap';
            s.framesLeft = Math.round(5 + Math.random() * 10);
        }
        return s.miniProb * ampDecay * envelope;
    }

    return 0;
}

let rsbBound = false;
export function bindRsbEvents(rowCount) {
    if (rsbBound) return; // idempotent — survives View Transition re-inits
    rsbBound = true;
    window.addEventListener('mousedown', (e) => { if (e.button === 0) rsbStart(rowCount); });
    window.addEventListener('mouseup', (e) => { if (e.button === 0) rsbRelease(); });
    window.addEventListener('touchstart', () => rsbStart(rowCount));
    window.addEventListener('touchend', rsbRelease);
}
