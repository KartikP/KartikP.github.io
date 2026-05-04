import { SPEED, LFP_HISTORY_MAX } from './config.js';
import { inputState, updateSmoothing } from './input.js';
import { rsbState, rsbTick } from './rsb.js';
import { learningTick, learningBurstEnvelope, learningIsBurstActive, learningState } from './learning.js';
import { hoverFiringRate, hoverIsActive } from './hover.js';
import { drawMinimap } from './minimap.js';

function angleToDeg(rad) {
    let deg = (rad * 180 / Math.PI) % 360;
    if (deg < 0) deg += 360;
    return deg.toFixed(0);
}

export function createRenderer(dom, neurons, getScheme) {
    const { rCtx, lCtx, mCtx, rCanvas, lCanvas } = dom;
    const { statStatus, statPop, statDir, pulseDot } = dom;
    const rowCount = neurons.length;

    let width, rHeight, lHeight;
    const lfpHistory = []; // entries: { spikes, active }
    let _lastScheme = null; // sentinel for per-scheme color cache rebuild

    // Simulated baseline value for pre-populating history so the LFP strip
    // appears full on first paint regardless of how wide the viewport is.
    const baselineMean = rowCount * 0.004;
    function synthBaseline() {
        const jitter = 0.6 + Math.random() * 0.8; // 0.6×–1.4× the mean
        return {
            spikes: Math.max(1, Math.round(baselineMean * jitter)),
            active: false,
        };
    }

    function ensureHistoryCoversWidth(targetPx) {
        // Each history entry renders as a SPEED-wide column; need enough to
        // cover the full canvas width plus a small cushion. Prepend synthetic
        // baseline entries if we're short (never truncate — real history stays).
        const needed = Math.ceil(targetPx / SPEED) + 8;
        while (lfpHistory.length < needed) {
            lfpHistory.unshift(synthBaseline());
        }
    }

    function resize() {
        // Read dimensions from the canvases themselves so we pick up the
        // final CSS-laid-out size rather than racing window.innerWidth
        // against flex layout on initial load.
        const w = rCanvas.offsetWidth || lCanvas.offsetWidth || window.innerWidth;
        const rH = rCanvas.offsetHeight;
        const lH = lCanvas.offsetHeight;
        if (!w || !rH || !lH) return; // layout not ready yet

        // Idempotent: if nothing changed, skip. Setting canvas.width clears
        // the drawing buffer, so re-running a no-op resize during the tick
        // loop would wipe accumulated raster history.
        if (w === width && rH === rHeight && lH === lHeight) return;

        width = w;
        rHeight = rCanvas.height = rH;
        lHeight = lCanvas.height = lH;
        rCanvas.width = w;
        lCanvas.width = w;

        // Make sure history covers the full canvas width for this viewport.
        ensureHistoryCoversWidth(width);

        rCtx.fillStyle = '#050507';
        rCtx.fillRect(0, 0, width, rHeight);
        lCtx.fillStyle = '#08080a';
        lCtx.fillRect(0, 0, width, lHeight);
    }

    // Window resize keeps working as a backstop.
    window.addEventListener('resize', resize);

    // ResizeObserver fires on the actual canvas boxes — covers the initial
    // layout settle as well as any later resize from CSS (container changes,
    // font loads, etc.).
    let ro = null;
    if (typeof ResizeObserver !== 'undefined') {
        ro = new ResizeObserver(() => resize());
        ro.observe(rCanvas);
        ro.observe(lCanvas);
    }

    resize();
    // If the first call raced layout, try again on the next frame.
    requestAnimationFrame(resize);

    let rafHandle = null;
    let stopped = false;

    function tick() {
        updateSmoothing();

        const isActive = inputState.movementIntensity > 1;
        const cs = getScheme();

        // Shift raster left
        rCtx.drawImage(rCanvas, -SPEED, 0);
        rCtx.fillStyle = 'rgba(5, 5, 7, 0.85)';
        rCtx.fillRect(width - SPEED, 0, SPEED, rHeight);

        // RSB state machine
        const rsbFiringProb = rsbTick(rowCount);

        // Learned / spontaneous-network-burst state — evolves with RSB count.
        learningTick(rowCount, rsbState.phase);
        const inLearnedBurst = learningIsBurstActive();
        // Hoist the time-only burst envelope out of the per-neuron loop.
        const learnEnvelope = inLearnedBurst
            ? learningState.burstBaseProb * learningBurstEnvelope()
            : 0;
        const learnMask = inLearnedBurst ? learningState.burstNeuronMask : null;

        // Per-neuron color strings are pure functions of the scheme + the
        // neuron's static properties (preferredAngle, baseAlpha). Caching
        // them once per scheme avoids ~200 template-literal allocations
        // and CSS color parses every burst frame. Rebuilds only when the
        // scheme reference actually changes.
        if (cs !== _lastScheme) {
            for (let i = 0; i < rowCount; i++) {
                const n = neurons[i];
                n._baselineColor = cs.baseline(n);
                n._evokedColor = cs.evoked(n);
            }
            _lastScheme = cs;
        }

        // Hover-evoked selective firing — the hovered element's preferred
        // angle drives the matching subpopulation of direction-tuned neurons,
        // at a higher intensity than mouse movement does.
        const inHover = hoverIsActive();

        // Spikes
        let totalSpikes = 0;
        const rowStep = rHeight / rowCount;
        rCtx.save();

        const movementIntensity = inputState.movementIntensity;
        const evokedScalar = movementIntensity * 0.003;

        neurons.forEach((n, i) => {
            // Skip the angular-tuning Math.exp entirely when there's no
            // movement-evoked drive — saves ~1000 exps per idle frame.
            let tuningEffect = 0;
            let evokedRate = 0;
            if (evokedScalar > 0) {
                let angularDist = Math.abs(inputState.smoothAngle - n.preferredAngle);
                if (angularDist > Math.PI) angularDist = Math.PI * 2 - angularDist;
                const sigma = n.tuningWidth;
                tuningEffect = Math.exp(-(angularDist * angularDist) / (2 * sigma * sigma));
                evokedRate = evokedScalar * tuningEffect;
            }

            const rsbRate = (rsbFiringProb > 0 && rsbState.neuronMask[i]) ? rsbFiringProb : 0;
            const learnRate = (learnMask && learnMask[i]) ? learnEnvelope : 0;
            const hoverRate = inHover ? hoverFiringRate(n) : 0;

            if (n.isPersistent) {
                const drive = evokedRate * 1.5;
                if (drive > n.residual) {
                    n.residual = drive;
                } else {
                    n.residual *= n.persistDecay;
                }
                if (n.residual < 0.0001) n.residual = 0;
            }

            let p;
            if (n.burstRemaining > 0) {
                p = n.burstProb;
                n.burstRemaining--;
            } else {
                p = n.baseExcitability + evokedRate + rsbRate + learnRate + hoverRate + (n.isPersistent ? n.residual : 0);
            }

            if (Math.random() < p) {
                if (n.isBursty && n.burstRemaining <= 0 && (evokedRate > 0.005 || n.residual > 0.005 || rsbRate > 0.05 || learnRate > 0.05)) {
                    n.burstRemaining = n.burstLen;
                }

                const isEvoked = (inputState.movementIntensity > 2 && tuningEffect > 0.7)
                    || n.burstRemaining > 0
                    || (n.isPersistent && n.residual > 0.003)
                    || rsbRate > 0
                    || learnRate > 0.01
                    || hoverRate > 0;
                const y = i * rowStep;

                rCtx.fillStyle = isEvoked ? n._evokedColor : n._baselineColor;
                rCtx.fillRect(width - SPEED, y, n.size, n.size);
                totalSpikes++;
            }
        });

        rCtx.restore();

        // UI updates
        const activePercent = ((totalSpikes / rowCount) * 100).toFixed(1);
        const rsbLabel = rsbState.phase === 'holding' ? 'RSB Init'
            : rsbState.phase === 'decaying' ? 'RSB Decay'
            : rsbState.phase === 'miniburst' ? 'RSB Mini'
            : null;
        statStatus.textContent = rsbLabel || (isActive ? 'Active' : 'Baseline');
        statStatus.style.color = rsbLabel ? cs.accent : (isActive ? cs.accent : cs.accentDim);
        statPop.textContent = activePercent + '%';
        statPop.style.color = isActive ? cs.accent : cs.accentDim;
        statDir.textContent = isActive ? angleToDeg(inputState.smoothAngle) + '\u00B0' : '\u2014';
        statDir.style.color = isActive ? cs.accent : cs.accentDim;
        pulseDot.style.background = isActive ? cs.dotActive : cs.dotBaseline;
        pulseDot.style.boxShadow = isActive ? cs.dotGlow : 'none';

        // LFP histogram — redraw from history each frame, y-axis normalized to visible max.
        lfpHistory.push({ spikes: totalSpikes, active: isActive });
        // Cap history to what the current canvas width actually needs,
        // plus a small cushion. This adapts as the window resizes.
        const historyCap = Math.ceil(width / SPEED) + 20;
        while (lfpHistory.length > historyCap) lfpHistory.shift();

        // Normalize to max of what's currently visible (floor at 1 to avoid /0).
        let visibleMax = 1;
        for (let i = 0; i < lfpHistory.length; i++) {
            if (lfpHistory[i].spikes > visibleMax) visibleMax = lfpHistory[i].spikes;
        }
        const topPad = 6; // keeps bars and glow from touching the top edge
        const availableH = Math.max(1, lHeight - topPad);

        // Clear and redraw all bars aligned to the right edge.
        lCtx.fillStyle = '#08080a';
        lCtx.fillRect(0, 0, width, lHeight);

        const n = lfpHistory.length;
        const minBarH = 1; // keep baseline visible even when dwarfed by a burst
        for (let i = 0; i < n; i++) {
            const entry = lfpHistory[i];
            const x = width - (n - i) * SPEED;
            if (x + SPEED < 0) continue;
            const rawH = (entry.spikes / visibleMax) * availableH;
            const barH = entry.spikes > 0 ? Math.max(minBarH, rawH) : 0;
            if (barH === 0) continue;
            lCtx.fillStyle = entry.active ? cs.lfpActive : cs.lfpBaseline;
            lCtx.fillRect(x, lHeight - barH, SPEED, barH);
        }

        // Glow on the newest bar only when active.
        if (isActive && totalSpikes > 0) {
            const latestBarH = (totalSpikes / visibleMax) * availableH;
            lCtx.shadowColor = cs.lfpGlow;
            lCtx.shadowBlur = 4;
            lCtx.fillStyle = cs.lfpActive;
            lCtx.fillRect(width - SPEED, lHeight - latestBarH, SPEED, latestBarH);
            lCtx.shadowBlur = 0;
        }

        // Minimap
        drawMinimap(mCtx, cs, inputState.smoothAngle, inputState.movementIntensity);

        if (!stopped) rafHandle = requestAnimationFrame(tick);
    }

    function stop() {
        stopped = true;
        if (rafHandle) cancelAnimationFrame(rafHandle);
        rafHandle = null;
        window.removeEventListener('resize', resize);
        if (ro) ro.disconnect();
    }

    return { tick, resize, stop };
}
