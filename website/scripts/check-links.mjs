import { readFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { parse } from 'parse5';

const directory = path.resolve('dist');
const pages = [];
async function visit(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) await visit(file);
    else if (entry.name.endsWith('.html')) pages.push(file);
  }
}
await visit(directory);
if (pages.length < 10) throw new Error('Documentation pages missing');
const parsed = new Map();
for (const file of pages) {
  const ids = new Set(), links = [];
  function walk(node) {
    for (const attr of node.attrs ?? []) {
      if (attr.name === 'id') ids.add(attr.value);
      if (attr.name === 'href' || attr.name === 'src') links.push(attr.value);
    }
    for (const child of node.childNodes ?? []) walk(child);
  }
  walk(parse(await readFile(file, 'utf8')));
  parsed.set(file, { ids, links });
}
const errors = [];
for (const [file, { links }] of parsed) {
  const location = new URL('/jevtpp/' + path.relative(directory, file), 'https://wiatrm.github.io');
  for (const href of links) {
    if (/^(data:|mailto:|javascript:)/.test(href)) continue;
    const url = new URL(href, location);
    if (url.origin !== location.origin) continue;
    // The generated error page's canonical URL intentionally returns 404.
    if (path.basename(file) === '404.html' && url.pathname === '/jevtpp/404/') continue;
    if (!url.pathname.startsWith('/jevtpp/')) { errors.push(`${file}: outside project base: ${href}`); continue; }
    let target = path.join(directory, decodeURIComponent(url.pathname.slice('/jevtpp/'.length)));
    try {
      if ((await stat(target)).isDirectory()) target = path.join(target, 'index.html');
      await stat(target);
      if (url.hash && parsed.has(target) && !parsed.get(target).ids.has(decodeURIComponent(url.hash.slice(1))))
        errors.push(`${file}: missing anchor: ${href}`);
    } catch { errors.push(`${file}: missing target: ${href}`); }
  }
}
if (errors.length) throw new Error(errors.join('\n'));
console.log(`Validated links, assets and anchors in ${pages.length} HTML pages.`);
