export const inputState = {
    lastX: 0,
    lastY: 0,
    movementAngle: 0,
    smoothAngle: 0,
    movementIntensity: 0,
    rawIntensity: 0,
};

function handleMove(dx, dy) {
    const s = inputState;
    s.rawIntensity = Math.min(Math.sqrt(dx * dx + dy * dy), 50);
    s.movementIntensity = Math.max(s.movementIntensity, s.rawIntensity * 0.5 + s.movementIntensity * 0.5);
    s.movementIntensity = Math.min(s.movementIntensity, 50);
    s.movementAngle = Math.atan2(dy, dx);
}

export function bindInputEvents() {
    window.addEventListener('mousemove', (e) => {
        const dx = e.clientX - inputState.lastX;
        const dy = e.clientY - inputState.lastY;
        handleMove(dx, dy);
        inputState.lastX = e.clientX;
        inputState.lastY = e.clientY;
    });

    window.addEventListener('touchmove', (e) => {
        const t = e.touches[0];
        const dx = t.clientX - inputState.lastX;
        const dy = t.clientY - inputState.lastY;
        handleMove(dx, dy);
        inputState.lastX = t.clientX;
        inputState.lastY = t.clientY;
    });
}

export function updateSmoothing() {
    const s = inputState;
    s.movementIntensity *= 0.95;
    if (s.movementIntensity < 0.1) s.movementIntensity = 0;

    let angleDiff = s.movementAngle - s.smoothAngle;
    if (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
    if (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
    s.smoothAngle += angleDiff * 0.35;
    if (s.smoothAngle > Math.PI) s.smoothAngle -= Math.PI * 2;
    if (s.smoothAngle < -Math.PI) s.smoothAngle += Math.PI * 2;
}
