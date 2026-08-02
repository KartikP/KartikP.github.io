# Homepage Raster Performance Design

## Goal

Reduce dropped or uneven frames on the homepage without changing its visible
appearance, interaction model, neural firing behavior, animation timing, or
navigation behavior. The work must preserve the 1,000-neuron simulation, the
two-pixel-per-frame raster motion, LFP normalization, cursor tuning, hover
responses, spontaneous bursts, and existing overlays.

## Architecture

The renderer will retain its single `requestAnimationFrame` loop and existing
simulation modules. Avoidable work will be removed from that loop first. The
hidden bridge elements will remain available for compatibility, but the
homepage renderer will not redraw the hidden minimap or update hidden status
readouts. Hover decay will be computed once per frame, while fixed
neuron-to-target tuning weights will be cached when a hover target changes.

The raster history will become a cyclic buffer. A double-width backing canvas
will contain two identical copies of each history column, allowing a clipped
viewport to display a contiguous, chronologically ordered window as the write
cursor wraps. Only the new two-pixel strip will be updated each frame. A small
edge accumulator will retain the current right-edge fade behavior before the
strip is copied into both halves of the cyclic backing canvas.

The LFP history will use a circular data buffer rather than `push` and
`shift`. Its maximum will be tracked with a monotonic queue. When the visible
maximum is unchanged, the existing strip will advance and only the newest bar
will be drawn. When normalization changes, the backing canvas will be rebuilt
so the result remains identical to the current full redraw.

## Data Flow and Compatibility

Input, RSB, learning, and hover events continue to update the same exported
state. Each animation frame advances those states once, calculates firing
probabilities for all neurons, and records the same spike count. Renderer
changes affect only how pixels and LFP history are stored.

Resize behavior remains intentionally reset-based, matching the current
canvas behavior. The visible canvas sizes still follow their CSS boxes at one
canvas pixel per CSS pixel. Astro view-transition teardown continues to stop
the frame loop, disconnect observers, and recreate fresh renderer state when
returning to the homepage.

The implementation will feature-detect any browser-specific optimization and
retain a direct Canvas 2D fallback where needed. It will not cap the frame
rate, lower canvas resolution, reduce neuron count, or remove visual effects.

## Verification

Run the production build and inspect the homepage at desktop and narrow
viewports. Verify baseline scrolling, cursor-driven tuning, link-hover bursts,
mouse-held RSB behavior, spontaneous bursts, LFP scaling, resize behavior,
dialog interaction, and navigation away from and back to the homepage.

Visual parity will be checked by comparing the structure and rendered page
before and after interaction. Runtime instrumentation will confirm that one
renderer loop is active, hidden bridge drawing no longer occurs, and cyclic
buffers update only the newly exposed strip during steady-state frames.
