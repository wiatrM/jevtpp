# Documentation site

Public site: <https://wiatrm.github.io/jevtpp/>.

Node 22.12+ and pnpm 10.30.3:

```sh
cd website
pnpm install --frozen-lockfile
pnpm dev
pnpm build
```

Author portable MDX in `content/`. Technical reference pages are sourced from
the existing repository `docs/*.md`; do not maintain duplicate copies.
`docs.json` is the Mintlify navigation/configuration source. `scripts/prepare.mjs`
produces ignored Starlight pages and a standalone `.mintlify/` content bundle.
The MDX deliberately uses Markdown and fenced code, not renderer-specific
components. A native Mintlify preview/deployment is separate and has not been
validated as part of the GitHub Pages build.

GitHub Pages uses Astro/Starlight, not the proprietary Mintlify renderer.
Mintlify's official offline export requires Enterprise:
<https://www.mintlify.com/docs/deploy/export>.

The build validates internal routes, anchors and assets under `/jevtpp/` and
generates a local search index. Pull requests build only; `main` deploys using
GitHub Actions. Enable **Settings → Pages → Source: GitHub Actions** if setting
up a fork; update `site` and `base` in `astro.config.mjs` and the canonical URLs
in the preparation/link-check scripts for a different repository.
