import { readFile, writeFile, mkdir, copyFile, readdir, unlink } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const repo = path.resolve(root, '..');
const references = {
  'system-one': 'docs/SYSTEM_ONE.md', laya: 'docs/LAYA.md', native: 'docs/NATIVE.md',
  observability: 'docs/OBSERVABILITY.md', performance: 'docs/PERFORMANCE.md',
  architecture: 'docs/ARCHITECTURE.md',
  batching: 'docs/BATCHING.md', precision: 'docs/PRECISION.md',
  remote: 'docs/REMOTE.md', asio: 'docs/ASIO.md', testing: 'docs/TESTING.md',
};
const config = JSON.parse(await readFile(path.join(root, 'docs.json'), 'utf8'));
const pages = config.navigation.groups.flatMap(group => group.pages);
if (new Set(pages).size !== pages.length) throw new Error('Duplicate navigation page');
const sourceFor = slug => references[slug] ?? `website/content/${slug}.mdx`;
const slugFor = Object.fromEntries(pages.map(slug => [sourceFor(slug), slug]));
for (const target of ['src/content/docs', '.mintlify']) {
  const directory = path.join(root, target);
  await mkdir(directory, { recursive: true });
  // Only remove stale generated page files in these dedicated output folders.
  for (const name of await readdir(directory)) {
    if (name.endsWith('.mdx') && !pages.includes(name.slice(0, -4))) await unlink(path.join(directory, name));
  }
  for (const slug of pages) {
    const source = sourceFor(slug);
    let content = await readFile(path.join(repo, source), 'utf8');
    if (!content.startsWith('---\n')) {
      const title = content.match(/^# (.+)\n/);
      if (!title) throw new Error(`Missing title in ${source}`);
      content = `---\ntitle: ${JSON.stringify(title[1])}\n---\n` + content.slice(title[0].length);
    }
    content = content.replace(/\]\(([^)]+)\)/g, (match, href) => {
      if (/^(https?:|mailto:|#)/.test(href)) return match;
      const [pathname, fragment] = href.split('#');
      const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(source), pathname));
      const localSlug = pages.includes(pathname) ? pathname : slugFor[resolved];
      if (localSlug) {
        const route = localSlug === 'index' ? '' : `${localSlug}/`;
        return `](${target === '.mintlify' ? '/' : '/jevtpp/'}${route}${fragment ? `#${fragment}` : ''})`;
      }
      return `](https://github.com/wiatrM/jevtpp/blob/main/${resolved}${fragment ? `#${fragment}` : ''})`;
    });
    if (target !== '.mintlify') content = content.replace(/^---\n/, `---\neditUrl: https://github.com/wiatrM/jevtpp/edit/main/${source}\n`);
    await writeFile(path.join(directory, `${slug}.mdx`), content);
  }
}
await copyFile(path.join(root, 'docs.json'), path.join(root, '.mintlify/docs.json'));
await copyFile(path.join(repo, 'docs/assets/jevtpp-logo-readme.png'), path.join(root, '.mintlify/logo.png'));
await copyFile(path.join(root, 'public/favicon.svg'), path.join(root, '.mintlify/favicon.svg'));
const overview = '# JevT++\n\n> Independent MIT-licensed C++20 library for typed model-backed routing, classification and scoring.\n\nApplication context becomes enum choices, boolean decisions or scores with explicit abstention. Local adapters run Laya through ONNX Runtime or native ggml; an opt-in remote backend implements the TypeSafe System One interface. This is not the proprietary Jev model or an official TypeSafe SDK. Remote tests use injected responses and loopback HTTP, not live-provider validation. Confidence is not measured accuracy. Features on main may be newer than the latest release.\n\n';
await writeFile(path.join(root, 'public/llms.txt'), overview + '## Documentation\n\n' + pages.map(slug =>
  `- [${slug}](https://wiatrm.github.io/jevtpp/${slug === 'index' ? '' : slug + '/'})`).join('\n') +
  '\n\n## Source and releases\n\n- [Source](https://github.com/wiatrM/jevtpp)\n- [Releases](https://github.com/wiatrM/jevtpp/releases)\n- [Full documentation text](https://wiatrm.github.io/jevtpp/llms-full.txt)\n');
const fullDocs = await Promise.all(pages.map(async slug => {
  const route = `https://wiatrm.github.io/jevtpp/${slug === 'index' ? '' : slug + '/'}`;
  const body = (await readFile(path.join(root, 'src/content/docs', `${slug}.mdx`), 'utf8')).replace(/^---\n[\s\S]*?\n---\n/, '');
  return `## ${slug}\n\nSource: ${route}\n\n${body}`;
}));
await writeFile(path.join(root, 'public/llms-full.txt'), overview + fullDocs.join('\n\n---\n\n'));
console.log(`Prepared ${pages.length} pages for Starlight and Mintlify.`);
