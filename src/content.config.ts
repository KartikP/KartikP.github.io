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

// Projects — selected work and prototypes.
const projects = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/projects" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    url: z.string().url().optional(),
    repo: z.string().url().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
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

export const collections = { posts, projects, publications };
