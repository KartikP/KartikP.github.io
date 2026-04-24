export const schemes = {
    greyscale: {
        baseline(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const l = 35 + t * 35;
            return `hsla(0, 0%, ${l}%, ${n.baseAlpha})`;
        },
        evoked(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const l = 55 + t * 25;
            return `hsla(0, 0%, ${l}%, 0.8)`;
        },
        ring(a) { const t = 1 - Math.abs((a / Math.PI) - 1); const l = 35 + t * 40; return `hsl(0, 0%, ${l}%)`; },
        lfpActive: '#aaa',
        lfpBaseline: '#444',
        lfpGlow: 'rgba(200, 200, 200, 0.3)',
        accent: '#ccc',
        accentDim: 'rgba(200, 200, 200, 0.4)',
        dotActive: '#fff',
        dotGlow: '0 0 8px #fff, 0 0 20px rgba(255,255,255,0.3)',
        dotBaseline: '#555',
    },
    green: {
        baseline(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const h = 100 + t * 80;
            return `hsla(${h}, 60%, 30%, ${n.baseAlpha})`;
        },
        evoked(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const h = 100 + t * 80;
            return `hsl(${h}, 65%, 55%)`;
        },
        ring(a) { const t = 1 - Math.abs((a / Math.PI) - 1); const h = 100 + t * 80; return `hsl(${h}, 65%, 50%)`; },
        lfpActive: '#4ade80',
        lfpBaseline: '#166534',
        lfpGlow: 'rgba(74, 222, 128, 0.4)',
        accent: '#4ade80',
        accentDim: 'rgba(74, 222, 128, 0.4)',
        dotActive: '#4ade80',
        dotGlow: '0 0 8px #4ade80, 0 0 20px rgba(74,222,128,0.3)',
        dotBaseline: '#166534',
    },
    bone: {
        baseline(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const h = 190 + t * 40;
            const l = 30 + t * 20;
            return `hsla(${h}, 15%, ${l}%, ${n.baseAlpha})`;
        },
        evoked(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const h = 190 + t * 40;
            const l = 50 + t * 20;
            return `hsl(${h}, 18%, ${l}%)`;
        },
        ring(a) { const t = 1 - Math.abs((a / Math.PI) - 1); const h = 190 + t * 40; const l = 35 + t * 30; return `hsl(${h}, 18%, ${l}%)`; },
        lfpActive: '#d6cfc2',
        lfpBaseline: '#4a5568',
        lfpGlow: 'rgba(214, 207, 194, 0.3)',
        accent: '#d6cfc2',
        accentDim: 'rgba(214, 207, 194, 0.4)',
        dotActive: '#d6cfc2',
        dotGlow: '0 0 8px #d6cfc2, 0 0 20px rgba(214,207,194,0.3)',
        dotBaseline: '#4a5568',
    },
    rainbow: {
        baseline(n) { return `hsla(${n.hue}, 30%, 35%, ${n.baseAlpha})`; },
        evoked(n) { return `hsla(${n.hue}, 85%, 75%, 0.95)`; },
        lfpActive: '#e0e0e0',
        lfpBaseline: '#555',
        lfpGlow: 'rgba(220, 220, 220, 0.3)',
        accent: '#e0e0e0',
        accentDim: 'rgba(220, 220, 220, 0.4)',
        dotActive: '#fff',
        dotGlow: '0 0 8px #fff, 0 0 20px rgba(255,255,255,0.3)',
        dotBaseline: '#555',
    },
    viridis: {
        baseline(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const r = Math.round(68 + t * 100);
            const g = Math.round(1 + t * 180);
            const b = Math.round(84 + (1 - t) * 80);
            return `rgba(${r}, ${g}, ${b}, ${n.baseAlpha})`;
        },
        evoked(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const r = Math.round(100 + t * 155);
            const g = Math.round(40 + t * 215);
            const b = Math.round(120 + (1 - t) * 50);
            return `rgba(${r}, ${g}, ${b}, 0.95)`;
        },
        lfpActive: '#b5de2b',
        lfpBaseline: '#31688e',
        lfpGlow: 'rgba(181, 222, 43, 0.4)',
        accent: '#b5de2b',
        accentDim: 'rgba(181, 222, 43, 0.4)',
        dotActive: '#b5de2b',
        dotGlow: '0 0 8px #b5de2b, 0 0 20px rgba(181,222,43,0.3)',
        dotBaseline: '#31688e',
    },
    magma: {
        baseline(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const r = Math.round(20 + t * 140);
            const g = Math.round(10 + t * 30);
            const b = Math.round(40 + t * 80);
            return `rgba(${r}, ${g}, ${b}, ${n.baseAlpha})`;
        },
        evoked(n) {
            const t = 1 - Math.abs((n.preferredAngle / Math.PI) - 1);
            const r = Math.round(120 + t * 135);
            const g = Math.round(30 + t * 180);
            const b = Math.round(80 + (1 - t) * 60);
            return `rgba(${r}, ${g}, ${b}, 0.95)`;
        },
        lfpActive: '#fcfdbf',
        lfpBaseline: '#721f81',
        lfpGlow: 'rgba(252, 253, 191, 0.3)',
        accent: '#fcfdbf',
        accentDim: 'rgba(252, 253, 191, 0.4)',
        dotActive: '#fcfdbf',
        dotGlow: '0 0 8px #fcfdbf, 0 0 20px rgba(252,253,191,0.3)',
        dotBaseline: '#721f81',
    }
};

export function colorForAngle(scheme, angle) {
    if (scheme.ring) {
        return scheme.ring(angle);
    }
    const fakeNeuron = {
        preferredAngle: angle,
        hue: (angle / (Math.PI * 2)) * 360,
        baseAlpha: 0.7
    };
    return scheme.evoked(fakeNeuron);
}
