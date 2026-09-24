import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { readFileSync } from 'node:fs';

const navigation = JSON.parse(readFileSync(new URL('./docs.json', import.meta.url))).navigation;
export default defineConfig({
  site: 'https://wiatrm.github.io',
  base: '/jevtpp',
  trailingSlash: 'always',
  integrations: [starlight({
    title: 'JevT++',
    description: 'Typed local decisions for C++20 applications. API, integration and measured performance.',
    logo: { src: '../docs/assets/jevtpp-logo-readme.png', replacesTitle: true },
    favicon: '/favicon.svg',
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/wiatrM/jevtpp' }],
    sidebar: navigation.groups.map(({ group, pages }) => ({
      label: group, items: pages.map(page => ({ slug: page === 'index' ? '' : page })),
    })),
    customCss: ['./src/styles/custom.css'],
  })],
});
