# Site Rebuild Design — 2026-04-21

## Goal

Replace the existing Jekyll/al-folio site at `kartikp.github.io` with a minimal, fast, modular, mobile-friendly static site. Build in `kartikp-site-v2/` alongside the old site; cut over when ready.

## Signature element

The site's visual identity is the neural raster from `KartikP/neural-raster` — a real-time Canvas visualization of a neural population firing in response to mouse movement, with directional tuning and Reverberating Super Bursts on click-hold. This is Kartik's own work and directly reflects his identity as a computational neuroscientist. It replaces the conventional "hero image" entirely.

## Visual system

**Landing page (`/`)** — dark portal
- Background: near-black (~#0A0A0A), full-bleed neural raster canvas (greyscale scheme — white spikes on black)
- Foreground: bold geometric sans-serif, large vertical nav list. Each item has a title, a small muted subtitle, and a right-aligned two-digit index (01/02/03/04). Left-aligned, ~15% from the left edge.
- No scrolling content beneath. The landing page IS the nav.
- Reference: `experiments.thisiswhitespace.com/cursor-nav` (layout only — we are not adopting the 3D cursor or debug panel)

**Content pages (`/overview`, `/posts`, `/projects`, `/publications`)** — light reading
- Background: cream/paper (~#F5F3EE)
- Single readable text face, reading-width constrained
- Simple top-left breadcrumb back to `/`
- No raster, no heavy animation — reading is a service, not a performance

## Nav sections (first pass)

Defined in a single config file (`src/config/site.ts`). Editing this file changes the landing page; index numbers auto-compute from array position.

1. **Overview** — a quiet entry point
2. **Posts** — notes, essays, fragments
3. **Projects** — selected work and prototypes
4. **Publications** — peer-reviewed work

## Tech stack: Astro

- Static HTML output, zero JS on content pages by default (no virtual DOM, no runtime framework)
- Markdown content collections with typed frontmatter (posts, projects, publications)
- Neural raster mounted as a plain `<script>` on the landing route only
- Deploys to GitHub Pages via one action

Why Astro over pure DIY: the alternative is writing markdown parsing, content-collection typing, and a dev server ourselves. The HTML output is identical either way.

## Content structure

```
src/
  config/
    site.ts              # nav, hero text, social links — edit here
  content/
    posts/*.md           # one file per post
    projects/*.md        # one file per project
    publications/*.md    # one file per publication
    config.ts            # Zod schemas for each collection
  pages/
    index.astro          # dark landing with raster
    overview.astro
    posts/
      index.astro        # list of posts
      [slug].astro       # individual post
    projects/
      index.astro
      [slug].astro
    publications/
      index.astro
      [slug].astro
  raster/                # vendored from KartikP/neural-raster
  layouts/
    DarkLayout.astro     # landing-only
    LightLayout.astro    # all content pages
  styles/
    global.css
```

## Mobile behavior

- Raster runs on desktop (mouse-driven by design)
- Touch devices: swap to a static greyscale PNG snapshot of the raster, or auto-animate without input (decide during implementation)
- `prefers-reduced-motion: reduce` → static snapshot for accessibility

## Out of scope (this iteration)

- Porting existing posts/projects/publications (user explicitly said skip)
- Password-protected `_locker` content from the old site
- Dark/light toggle on content pages (landing is dark, content is light, fixed)
- Search, tag clouds, RSS (can add later)

## Success criteria

- Lighthouse mobile performance ≥ 95
- Adding a new post = drop one `.md` file into `src/content/posts/`
- Adding a new nav section = add one entry to `src/config/site.ts` + one page file
- Landing page feels visually distinct and signals "computational neuroscientist" without requiring any copy to explain it
