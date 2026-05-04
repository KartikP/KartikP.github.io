import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

// Posts — notes, essays, fragments.
const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
    // Opt-in embeds. Set to true in a post's frontmatter to inject the
    // corresponding viewer/library + init script on that post only.
    brainbrowser: z.boolean().default(false),
  }),
});

// Projects — selected work and prototypes. Routed through CaseStudyLayout
// just like /work entries. Markdown projects render their content inside
// a .prose wrapper; MDX projects can additionally use case-study and
// sciviz components inline.
const projects = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/projects" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    summary: z.string().optional(),     // long-form hero paragraph; falls back to description
    accent: z.string().default("#404040"),
    url: z.string().url().optional(),
    repo: z.string().url().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
    // Optional case-study meta — render a MetaBar if any of these is set.
    role: z.string().optional(),
    team: z.string().optional(),
    timeline: z.string().optional(),
    skills: z.array(z.string()).default([]),
    // Hero artwork — either a static image or a named live component.
    heroImage: z.string().optional(),
    heroImageAlt: z.string().optional(),
    heroComponent: z.enum(["RasterMini"]).optional(),
  }),
});

// Work — long-form case studies authored in MDX. Richer than projects:
// per-study accent colour, hero artwork, and structured meta (role, team,
// timeline, skills). Each entry routes at /work/<slug> and uses the
// CaseStudyLayout with TOC sidebar and component kit.
const work = defineCollection({
  loader: glob({ pattern: "**/*.mdx", base: "./src/content/work" }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    date: z.coerce.date(),
    accent: z.string().default("#c5663a"),
    accentSoft: z.string().optional(), // hero gradient (defaults to accent at low opacity)
    tags: z.array(z.string()).default([]),
    role: z.string().optional(),
    team: z.string().optional(),
    timeline: z.string().optional(),
    skills: z.array(z.string()).default([]),
    repo: z.string().url().optional(),
    url: z.string().url().optional(),
    draft: z.boolean().default(false),
  }),
});

// Publications — peer-reviewed work.
const publications = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/publications" }),
  schema: z.object({
    title: z.string(),
    authors: z.string(),
    venue: z.string(),
    year: z.number(),
    doi: z.string().optional(),
    url: z.string().url().optional(),
    pdf: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { posts, projects, publications, work };
