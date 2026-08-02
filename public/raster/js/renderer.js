import { SPEED } from './config.js';
import { inputState, updateSmoothing } from './input.js';
import { rsbState, rsbTick } from './rsb.js';
import { learningTick, learningBurstEnvelope, learningIsBurstActive, learningState } from './learning.js';
import { prepareHoverFrame, hoverFiringRate } from './hover.js';

const RASTER_BG = '#050507';
const LFP_BG = '#08080a';
const LFP_GLOW_WIDTH = 16;

export function createRenderer(dom, neurons, getScheme) {
    const { rCtx, lCtx, gCtx, rCanvas, lCanvas, glowCanvas } = dom;
    const rowCount = neurons.length;

    let width = 0;
    let rHeight = 0;
    let lHeight = 0;
    let rasterWriteX = 0;
    let _lastScheme = null;

    // The visible raster is a viewport onto a double-width cyclic canvas.
    // This two-pixel strip retains the right-edge fade from the original
    // renderer before being copied into both halves of the cyclic buffer.
    const edgeCanvas = document.createElement('canvas');
    const edgeCtx = edgeCanvas.getContext('2d', { alpha: false });

    // LFP history uses fixed-capacity typed arrays. A monotonic queue tracks
    // the maximum without rescanning the full history every frame.
    let historySpikes = new Uint16Array(0);
    let historyActive = new Uint8Array(0);
    let historySerials = new Float64Array(0);
    let historyCapacity = 0;
    let historyStart = 0;
    let historyLength = 0;
    let nextHistorySerial = 0;
    let maxSerials = [];
    let maxValues = [];
    let maxHead = 0;
    let renderedLfpMax = 0;
    let renderedLfpScheme = null;
    let lfpNeedsRedraw = true;

    const baselineMean = rowCount * 0.004;

    function synthBaseline() {
        const jitter = 0.6 + Math.random() * 0.8;
        return {
            spikes: Math.max(1, Math.round(baselineMean * jitter)),
            active: false,
        };
    }

    function historyIndex(logicalIndex) {
        return (historyStart + logicalIndex) % historyCapacity;
    }

    function snapshotHistory() {
        const entries = new Array(historyLength);
        for (let i = 0; i < historyLength; i++) {
            const index = historyIndex(i);
            entries[i] = {
                spikes: historySpikes[index],
                active: historyActive[index] === 1,
            };
        }
        return entries;
    }

    function compactMaxQueue() {
        if (maxHead > 256 && maxHead * 2 > maxValues.length) {
            maxSerials = maxSerials.slice(maxHead);
            maxValues = maxValues.slice(maxHead);
            maxHead = 0;
        }
    }

    function appendHistory(spikes, active) {
        if (historyCapacity === 0) return;

        if (historyLength === historyCapacity) {
            const evictedSerial = historySerials[historyStart];
            historyStart = (historyStart + 1) % historyCapacity;
            historyLength--;
            while (maxHead < maxSerials.length && maxSerials[maxHead] <= evictedSerial) {
                maxHead++;
            }
        }

        const index = historyIndex(historyLength);
        const serial = nextHistorySerial++;
        historySpikes[index] = spikes;
        historyActive[index] = active ? 1 : 0;
        historySerials[index] = serial;
        historyLength++;

        while (maxValues.length > maxHead && maxValues[maxValues.length - 1] <= spikes) {
            maxValues.pop();
            maxSerials.pop();
        }
        maxSerials.push(serial);
        maxValues.push(spikes);
        compactMaxQueue();
    }

    function configureHistory(targetPx) {
        const previous = snapshotHistory();
        const needed = Math.ceil(targetPx / SPEED) + 8;
        const capacity = Math.ceil(targetPx / SPEED) + 20;
        let entries = previous.length > capacity
            ? previous.slice(previous.length - capacity)
            : previous;

        while (entries.length < needed) {
            entries.unshift(synthBaseline());
        }

        historySpikes = new Uint16Array(capacity);
        historyActive = new Uint8Array(capacity);
        historySerials = new Float64Array(capacity);
        historyCapacity = capacity;
        historyStart = 0;
        historyLength = 0;
        nextHistorySerial = 0;
        maxSerials = [];
        maxValues = [];
        maxHead = 0;

        for (let i = 0; i < entries.length; i++) {
            appendHistory(entries[i].spikes, entries[i].active);
        }
        lfpNeedsRedraw = true;
    }

    function historyMax() {
        return Math.max(1, maxHead < maxValues.length ? maxValues[maxHead] : 0);
    }

    function drawLfpBar(ctx, x, spikes, active, cs, visibleMax) {
        const topPad = 6;
        const availableH = Math.max(1, lHeight - topPad);
        const rawH = (spikes / visibleMax) * availableH;
        const barH = spikes > 0 ? Math.max(1, rawH) : 0;
        if (barH === 0) return;
        ctx.fillStyle = active ? cs.lfpActive : cs.lfpBaseline;
        ctx.fillRect(x, lHeight - barH, SPEED, barH);
    }

    function redrawLfp(cs, visibleMax) {
        lCtx.fillStyle = LFP_BG;
        lCtx.fillRect(0, 0, width, lHeight);

        for (let i = 0; i < historyLength; i++) {
            const x = width - (historyLength - i) * SPEED;
            if (x + SPEED < 0) continue;
            const index = historyIndex(i);
            drawLfpBar(
                lCtx,
                x,
                historySpikes[index],
                historyActive[index] === 1,
                cs,
                visibleMax,
            );
        }

        renderedLfpMax = visibleMax;
        renderedLfpScheme = cs;
        lfpNeedsRedraw = false;
    }

    function advanceLfp(spikes, active, cs) {
        appendHistory(spikes, active);
        const visibleMax = historyMax();

        if (lfpNeedsRedraw || visibleMax !== renderedLfpMax || cs !== renderedLfpScheme) {
            redrawLfp(cs, visibleMax);
        } else {
            // With stable normalization, every existing bar simply advances
            // left by SPEED and only the newly exposed strip needs drawing.
            lCtx.drawImage(lCanvas, -SPEED, 0);
            lCtx.fillStyle = LFP_BG;
            lCtx.fillRect(width - SPEED, 0, SPEED, lHeight);
            drawLfpBar(lCtx, width - SPEED, spikes, active, cs, visibleMax);
        }

        // Draw the active glow on a small transparent overlay so it never
        // becomes part of the scrolling LFP history.
        gCtx.clearRect(0, 0, LFP_GLOW_WIDTH, lHeight);
        if (active && spikes > 0) {
            const availableH = Math.max(1, lHeight - 6);
            const latestBarH = (spikes / visibleMax) * availableH;
            gCtx.shadowColor = cs.lfpGlow;
            gCtx.shadowBlur = 4;
            gCtx.fillStyle = cs.lfpActive;
            gCtx.fillRect(
                LFP_GLOW_WIDTH - SPEED,
                lHeight - latestBarH,
                SPEED,
                latestBarH,
            );
            gCtx.shadowBlur = 0;
        }
    }

    function resetRasterBuffer() {
        edgeCanvas.width = SPEED;
        edgeCanvas.height = rHeight;
        edgeCtx.fillStyle = RASTER_BG;
        edgeCtx.fillRect(0, 0, SPEED, rHeight);

        rCtx.fillStyle = RASTER_BG;
        rCtx.fillRect(0, 0, width * 2, rHeight);
        rasterWriteX = 0;
        rCanvas.style.transform = 'translate3d(0, 0, 0)';
    }

    function copyRasterStrip(writeX) {
        const firstWidth = Math.min(SPEED, width - writeX);
        rCtx.drawImage(
            edgeCanvas,
            0,
            0,
            firstWidth,
            rHeight,
            writeX,
            0,
            firstWidth,
            rHeight,
        );
        rCtx.drawImage(
            edgeCanvas,
            0,
            0,
            firstWidth,
            rHeight,
            writeX + width,
            0,
            firstWidth,
            rHeight,
        );

        if (firstWidth < SPEED) {
            const wrappedWidth = SPEED - firstWidth;
            rCtx.drawImage(
                edgeCanvas,
                firstWidth,
                0,
                wrappedWidth,
                rHeight,
                0,
                0,
                wrappedWidth,
                rHeight,
            );
            rCtx.drawImage(
                edgeCanvas,
                firstWidth,
                0,
                wrappedWidth,
                rHeight,
                width,
                0,
                wrappedWidth,
                rHeight,
            );
        }

        const viewportStart = (writeX + SPEED) % width;
        rCanvas.style.transform = `translate3d(${-viewportStart}px, 0, 0)`;
        rasterWriteX = viewportStart;
    }

    function resize() {
        // The raster canvas itself is double width, so dimensions come from
        // its clipped viewport rather than from the canvas element.
        const viewport = rCanvas.parentElement;
        const w = viewport.offsetWidth || window.innerWidth;
        const rH = viewport.offsetHeight;
        const lH = lCanvas.offsetHeight;
        if (!w || !rH || !lH) return;

        if (w === width && rH === rHeight && lH === lHeight) return;

        width = w;
        rHeight = rH;
        lHeight = lH;
        rCanvas.width = width * 2;
        rCanvas.height = rHeight;
        lCanvas.width = width;
        lCanvas.height = lHeight;
        glowCanvas.width = LFP_GLOW_WIDTH;
        glowCanvas.height = lHeight;

        configureHistory(width);
        resetRasterBuffer();

        lCtx.fillStyle = LFP_BG;
        lCtx.fillRect(0, 0, width, lHeight);
        gCtx.clearRect(0, 0, LFP_GLOW_WIDTH, lHeight);
        renderedLfpMax = 0;
        renderedLfpScheme = null;
        lfpNeedsRedraw = true;
    }

    window.addEventListener('resize', resize);

    let ro = null;
    if (typeof ResizeObserver !== 'undefined') {
        ro = new ResizeObserver(() => resize());
        ro.observe(rCanvas.parentElement);
        ro.observe(lCanvas);
    }

    resize();
    requestAnimationFrame(resize);

    let rafHandle = null;
    let stopped = false;

    function tick() {
        updateSmoothing();

        if (!width) {
            if (!stopped) rafHandle = requestAnimationFrame(tick);
            return;
        }

        const isActive = inputState.movementIntensity > 1;
        const cs = getScheme();

        // Preserve the original right-edge phosphor fade in a two-pixel
        // accumulator. The completed strip is committed after spike drawing.
        edgeCtx.fillStyle = 'rgba(5, 5, 7, 0.85)';
        edgeCtx.fillRect(0, 0, SPEED, rHeight);

        const rsbFiringProb = rsbTick(rowCount);

        learningTick(rowCount, rsbState.phase);
        const inLearnedBurst = learningIsBurstActive();
        const learnEnvelope = inLearnedBurst
            ? learningState.burstBaseProb * learningBurstEnvelope()
            : 0;
        const learnMask = inLearnedBurst ? learningState.burstNeuronMask : null;

        if (cs !== _lastScheme) {
            for (let i = 0; i < rowCount; i++) {
                const n = neurons[i];
                n._baselineColor = cs.baseline(n);
                n._evokedColor = cs.evoked(n);
            }
            _lastScheme = cs;
        }

        const hoverScalar = prepareHoverFrame(neurons);
        const movementIntensity = inputState.movementIntensity;
        const evokedScalar = movementIntensity * 0.003;
        const rowStep = rHeight / rowCount;
        let totalSpikes = 0;

        for (let i = 0; i < rowCount; i++) {
            const n = neurons[i];
            let tuningEffect = 0;
            let evokedRate = 0;
            if (evokedScalar > 0) {
                let angularDist = Math.abs(inputState.smoothAngle - n.preferredAngle);
                if (angularDist > Math.PI) angularDist = Math.PI * 2 - angularDist;
                const sigma = n.tuningWidth;
                tuningEffect = Math.exp(-(angularDist * angularDist) / (2 * sigma * sigma));
                evokedRate = evokedScalar * tuningEffect;
            }

            const rsbRate = (rsbFiringProb > 0 && rsbState.neuronMask[i])
                ? rsbFiringProb
                : 0;
            const learnRate = (learnMask && learnMask[i]) ? learnEnvelope : 0;
            const hoverRate = hoverFiringRate(i, hoverScalar);

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
                p = n.baseExcitability
                    + evokedRate
                    + rsbRate
                    + learnRate
                    + hoverRate
                    + (n.isPersistent ? n.residual : 0);
            }

            if (Math.random() < p) {
                if (
                    n.isBursty
                    && n.burstRemaining <= 0
                    && (
                        evokedRate > 0.005
                        || n.residual > 0.005
                        || rsbRate > 0.05
                        || learnRate > 0.05
                    )
                ) {
                    n.burstRemaining = n.burstLen;
                }

                const isEvoked = (movementIntensity > 2 && tuningEffect > 0.7)
                    || n.burstRemaining > 0
                    || (n.isPersistent && n.residual > 0.003)
                    || rsbRate > 0
                    || learnRate > 0.01
                    || hoverRate > 0;

                edgeCtx.fillStyle = isEvoked ? n._evokedColor : n._baselineColor;
                edgeCtx.fillRect(0, i * rowStep, n.size, n.size);
                totalSpikes++;
            }
        }

        copyRasterStrip(rasterWriteX);
        advanceLfp(totalSpikes, isActive, cs);

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
