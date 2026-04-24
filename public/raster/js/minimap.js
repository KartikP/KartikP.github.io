import { colorForAngle } from './schemes.js';

const SIZE = 110;
const CX = SIZE / 2;
const CY = SIZE / 2;
const OUTER_R = 48;
const INNER_R = 30;

export function drawMinimap(ctx, scheme, angle, intensity) {
    ctx.clearRect(0, 0, SIZE, SIZE);

    // Color ring
    const steps = 72;
    const wedge = (Math.PI * 2) / steps;
    for (let i = 0; i < steps; i++) {
        const a = (i / steps) * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(CX, CY, OUTER_R, a - wedge / 2, a + wedge / 2 + 0.01);
        ctx.arc(CX, CY, INNER_R, a + wedge / 2 + 0.01, a - wedge / 2, true);
        ctx.closePath();
        ctx.fillStyle = colorForAngle(scheme, a);
        ctx.fill();
    }

    // Cardinal labels
    ctx.font = '8px SF Mono, Fira Code, Courier New, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(200, 214, 229, 0.4)';
    const labelR = OUTER_R + 7;
    ctx.fillText('0\u00B0', CX + labelR, CY);
    ctx.fillText('90\u00B0', CX, CY + labelR);
    ctx.fillText('180\u00B0', CX - labelR, CY);
    ctx.fillText('270\u00B0', CX, CY - labelR);

    // Direction arrow
    const arrowLen = intensity > 1 ? INNER_R - 4 : 0;
    if (arrowLen > 0) {
        const ax = CX + Math.cos(angle) * arrowLen;
        const ay = CY + Math.sin(angle) * arrowLen;

        ctx.beginPath();
        ctx.moveTo(CX, CY);
        ctx.lineTo(ax, ay);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(255,255,255,0.6)';
        ctx.shadowBlur = 6;
        ctx.stroke();
        ctx.shadowBlur = 0;

        const headLen = 6;
        const headAngle = 0.5;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - Math.cos(angle - headAngle) * headLen,
                    ay - Math.sin(angle - headAngle) * headLen);
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - Math.cos(angle + headAngle) * headLen,
                    ay - Math.sin(angle + headAngle) * headLen);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(CX, CY, OUTER_R + 2, angle - 0.15, angle + 0.15);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'rgba(255,255,255,0.5)';
        ctx.shadowBlur = 6;
        ctx.stroke();
        ctx.shadowBlur = 0;
    } else {
        ctx.beginPath();
        ctx.arc(CX, CY, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(200, 200, 200, 0.3)';
        ctx.fill();
    }
}
