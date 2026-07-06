// Single source of truth for site-wide config.
// Edit this file to change the hero, nav sections, or social links.

export const site = {
  name: "Kartik Pradeepan",
  tagline: "Computational neuroscientist",
  hero: "Biology solved intelligence long before silicon did. Building software and hardware to learn how to replicate it.",
  // Meta description for search engines (the Google snippet). Mirrors the About page,
  // not the visual hero tagline. ~160 chars is the display sweet spot.
  description:
    "Kartik Pradeepan is a research scientist and engineer working across neuroscience, AI, and research infrastructure, building software to evaluate how closely AI matches biological intelligence.",
  email: "kartikspradeepan@gmail.com",
  social: {
    github: "https://github.com/KartikP",
  },
} as const;

export type NavItem = {
  title: string;
  subtitle: string;
  href: string;
};

// The landing page nav. Reorder, rename, add, or remove entries freely.
// Index numbers (01/02/…) auto-compute from array position — no manual bookkeeping.
export const nav: NavItem[] = [
  { title: "About",        subtitle: "The short version",            href: "/about" },
  { title: "Posts",        subtitle: "Notes, essays, fragments",     href: "/posts" },
  { title: "Projects",     subtitle: "Selected work and prototypes", href: "/projects" },
  { title: "Publications", subtitle: "Peer-reviewed work",           href: "/publications" },
];
