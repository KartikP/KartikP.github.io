import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://kartikp.github.io',
  output: 'static',
  trailingSlash: 'ignore',
  build: {
    inlineStylesheets: 'auto',
  },
  vite: {
    build: {
      cssCodeSplit: true,
    },
  },
});
