import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const REPOSITORY_ROOT = process.cwd();
const BOOK_ROOTS = [
    'docs/docs/en/src',
    'docs/docs/zh/src',
    'docs/blog/en/src',
    'docs/blog/zh/src'
];
const DOCS_ROOTS = ['docs/docs/en/src', 'docs/docs/zh/src'];
const errors = [];

function walk(directory) {
    return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
        const entryPath = path.join(directory, entry.name);
        return entry.isDirectory() ? walk(entryPath) : [entryPath];
    });
}

function markdownFiles(directory) {
    return walk(directory).filter(file => file.endsWith('.md'));
}

function markdownLinesOutsideFences(file) {
    const lines = readFileSync(file, 'utf8').split(/\r?\n/);
    let fenceMarker = null;

    return lines.map((line, index) => {
        const fence = line.match(/^\s*(`{3,}|~{3,})/);
        if (fence !== null) {
            if (fenceMarker === null) {
                fenceMarker = fence[1][0];
            } else if (fence[1][0] === fenceMarker) {
                fenceMarker = null;
            }
            return { line: '', lineNumber: index + 1 };
        }
        return { line: fenceMarker === null ? line : '', lineNumber: index + 1 };
    });
}

function headingSlug(text) {
    return text
        .replace(/<[^>]+>/g, '')
        .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
        .replace(/`([^`]*)`/g, '$1')
        .trim()
        .toLowerCase()
        .replace(/\s/g, '-')
        .replace(/[^\p{Letter}\p{Number}_-]/gu, '');
}

function anchorsFor(file) {
    const anchors = new Set();
    const occurrences = new Map();

    for (const { line } of markdownLinesOutsideFences(file)) {
        const explicitAnchors = line.matchAll(/<(?:a\s+[^>]*id|[^>]+\s+id)=["']([^"']+)["'][^>]*>/g);
        for (const match of explicitAnchors) {
            anchors.add(match[1]);
        }

        const heading = line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/);
        if (heading === null) {
            continue;
        }
        const base = headingSlug(heading[1]);
        const occurrence = occurrences.get(base) ?? 0;
        occurrences.set(base, occurrence + 1);
        anchors.add(occurrence === 0 ? base : `${base}-${occurrence}`);
    }
    return anchors;
}

function splitLinkTarget(rawTarget) {
    let target = rawTarget.trim();
    if (target.startsWith('<') && target.includes('>')) {
        target = target.slice(1, target.indexOf('>'));
    } else {
        target = target.split(/\s+["']/)[0];
    }
    const hashIndex = target.indexOf('#');
    const filePart = hashIndex === -1 ? target : target.slice(0, hashIndex);
    const anchor = hashIndex === -1 ? '' : decodeURIComponent(target.slice(hashIndex + 1));
    return { filePart: decodeURIComponent(filePart.split('?')[0]), anchor };
}

function resolveMarkdownTarget(sourceFile, filePart) {
    let target = filePart === '' ? sourceFile : path.resolve(path.dirname(sourceFile), filePart);
    if (existsSync(target) && statSync(target).isDirectory()) {
        target = path.join(target, 'README.md');
    }
    return target;
}

function checkMarkdownLinks() {
    const anchorCache = new Map();
    for (const root of BOOK_ROOTS) {
        for (const file of markdownFiles(root)) {
            for (const { line, lineNumber } of markdownLinesOutsideFences(file)) {
                const withoutInlineCode = line.replace(/`[^`]*`/g, '');
                for (const match of withoutInlineCode.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)) {
                    const rawTarget = match[1];
                    if (/^(?:https?:|mailto:|data:)/.test(rawTarget) || rawTarget.startsWith('/')) {
                        continue;
                    }
                    const { filePart, anchor } = splitLinkTarget(rawTarget);
                    if (path.basename(file) !== 'SUMMARY.md' && filePart.endsWith('README.md')) {
                        errors.push(
                            `${file}:${lineNumber}: inline README.md link renders as missing ` +
                                'README.html; link to the directory instead'
                        );
                    }
                    const target = resolveMarkdownTarget(file, filePart);
                    if (!existsSync(target)) {
                        errors.push(`${file}:${lineNumber}: missing local link target ${rawTarget}`);
                        continue;
                    }
                    if (anchor === '' || !target.endsWith('.md')) {
                        continue;
                    }
                    if (!anchorCache.has(target)) {
                        anchorCache.set(target, anchorsFor(target));
                    }
                    if (!anchorCache.get(target).has(anchor)) {
                        errors.push(`${file}:${lineNumber}: missing anchor ${rawTarget}`);
                    }
                }
            }
        }
    }
}

function checkBookNavigation() {
    for (const root of BOOK_ROOTS) {
        const summaryPath = path.join(root, 'SUMMARY.md');
        const summary = readFileSync(summaryPath, 'utf8');
        const listedPages = new Set(
            [...summary.matchAll(/\]\(([^)#]+\.md)(?:#[^)]+)?\)/g)].map(match =>
                path.normalize(match[1])
            )
        );
        const pages = markdownFiles(root)
            .map(file => path.relative(root, file))
            .filter(file => file !== 'SUMMARY.md');

        for (const page of pages) {
            if (!listedPages.has(page)) {
                errors.push(`${summaryPath}: page is missing from navigation: ${page}`);
            }
        }
        for (const page of listedPages) {
            if (!existsSync(path.join(root, page))) {
                errors.push(`${summaryPath}: navigation target does not exist: ${page}`);
            }
        }
    }
}

function checkLanguageParity() {
    const [englishRoot, chineseRoot] = DOCS_ROOTS;
    const englishFiles = new Set(walk(englishRoot).map(file => path.relative(englishRoot, file)));
    const chineseFiles = new Set(walk(chineseRoot).map(file => path.relative(chineseRoot, file)));

    for (const file of englishFiles) {
        if (!chineseFiles.has(file)) {
            errors.push(`${englishRoot}: missing Chinese counterpart: ${file}`);
        }
    }
    for (const file of chineseFiles) {
        if (!englishFiles.has(file)) {
            errors.push(`${chineseRoot}: missing English counterpart: ${file}`);
        }
    }
}

function checkReferencedRepositoryPaths() {
    const pathPattern =
        /`((?:examples|include|src|tools|python|python_bindings|tests|scripts|docs)\/[A-Za-z0-9_./+-]+\.(?:h|hpp|cpp|c|py|ts|js|mjs|md|yml|yaml|json|toml|sh|cmake))`/g;

    for (const root of BOOK_ROOTS) {
        for (const file of markdownFiles(root)) {
            const lines = readFileSync(file, 'utf8').split(/\r?\n/);
            for (const [index, line] of lines.entries()) {
                for (const match of line.matchAll(pathPattern)) {
                    if (!existsSync(path.join(REPOSITORY_ROOT, match[1]))) {
                        errors.push(`${file}:${index + 1}: missing repository path ${match[1]}`);
                    }
                }
            }
        }
    }
}

checkLanguageParity();
checkBookNavigation();
checkMarkdownLinks();
checkReferencedRepositoryPaths();

if (errors.length > 0) {
    console.error(`Documentation validation failed with ${errors.length} error(s):`);
    for (const error of errors) {
        console.error(`- ${error}`);
    }
    process.exitCode = 1;
} else {
    console.log('Documentation structure, local links, anchors, and repository paths are valid.');
}
