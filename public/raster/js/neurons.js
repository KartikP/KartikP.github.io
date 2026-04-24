export function createNeurons(count) {
    return Array.from({ length: count }, () => {
        const isHighFire = Math.random() > 0.96;
        const isBursty = Math.random() < 0.15;
        const isPersistent = Math.random() < 0.07;
        const preferredAngle = Math.random() * Math.PI * 2;
        const hue = (preferredAngle / (Math.PI * 2)) * 360;
        return {
            preferredAngle,
            tuningWidth: 0.2 + Math.random() * 0.5,
            baseExcitability: isHighFire ? 0.02 : 0.003,
            hue,
            baseAlpha: isHighFire ? 0.7 : 0.4,
            size: isHighFire ? 2.5 : 1.5,
            isHighFire,
            isBursty,
            burstLen: isBursty ? Math.floor(3 + Math.random() * 6) : 0,
            burstProb: isBursty ? 0.6 + Math.random() * 0.3 : 0,
            burstRemaining: 0,
            isPersistent,
            persistDecay: isPersistent ? 0.94 + Math.random() * 0.04 : 0,
            residual: 0
        };
    });
}
