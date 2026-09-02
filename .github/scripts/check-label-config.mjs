#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const manifestUrl = new URL('../labels.json', import.meta.url);
const mergifyUrl = new URL('../mergify.yml', import.meta.url);

const labels = JSON.parse(await readFile(manifestUrl, 'utf8'));
const mergify = await readFile(mergifyUrl, 'utf8');

assert(Array.isArray(labels), 'labels.json must contain an array');

const names = new Set();
for (const [index, label] of labels.entries()) {
    const context = `labels.json entry ${index}`;
    assert(label !== null && typeof label === 'object' && !Array.isArray(label), `${context} must be an object`);
    assert.equal(typeof label.name, 'string', `${context}.name must be a string`);
    assert.equal(typeof label.color, 'string', `${context}.color must be a string`);
    assert.equal(typeof label.description, 'string', `${context}.description must be a string`);
    assert.match(label.name, /^(?:area|module)\/[a-z0-9-]+$/, `invalid managed label: ${label.name}`);
    assert(!names.has(label.name), `duplicate managed label: ${label.name}`);
    assert.match(label.color, /^[0-9A-F]{6}$/, `invalid color for ${label.name}`);
    assert(label.description.trim().length > 0, `missing description for ${label.name}`);
    assert(label.legacy === undefined || label.legacy === true, `invalid legacy flag for ${label.name}`);
    names.add(label.name);
}

const referencedNames = new Set(
    [...mergify.matchAll(/^\s+- ((?:area|module)\/[a-z0-9-]+)\s*$/gm)].map((match) => match[1])
);

const missingFromManifest = [...referencedNames].filter((name) => !names.has(name));
const activeNames = labels.filter((label) => !label.legacy).map((label) => label.name);
const missingFromMergify = activeNames.filter((name) => !referencedNames.has(name));
const referencedLegacyNames = labels
    .filter((label) => label.legacy && referencedNames.has(label.name))
    .map((label) => label.name);

assert.deepEqual(missingFromManifest, [], `Mergify labels missing from manifest: ${missingFromManifest}`);
assert.deepEqual(missingFromMergify, [], `manifest labels unused by Mergify: ${missingFromMergify}`);
assert.deepEqual(referencedLegacyNames, [], `Mergify still uses legacy labels: ${referencedLegacyNames}`);

const remoteIndex = process.argv.indexOf('--remote');
if (remoteIndex !== -1) {
    const repository = process.argv[remoteIndex + 1] ?? process.env.GITHUB_REPOSITORY;
    assert(repository, '--remote requires an owner/repository argument or GITHUB_REPOSITORY');

    const headers = {
        Accept: 'application/vnd.github+json',
        'User-Agent': 'vsag-label-config-check',
        'X-GitHub-Api-Version': '2022-11-28'
    };
    if (process.env.GITHUB_TOKEN) {
        headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
    }

    const remoteLabels = [];
    for (let page = 1; ; page += 1) {
        const response = await fetch(
            `https://api.github.com/repos/${repository}/labels?per_page=100&page=${page}`,
            { headers, signal: AbortSignal.timeout(30_000) }
        );
        assert(response.ok, `GitHub labels request failed: ${response.status} ${response.statusText}`);

        const pageLabels = await response.json();
        remoteLabels.push(...pageLabels);
        if (pageLabels.length < 100) {
            break;
        }
    }

    const expectedByName = new Map(labels.map((label) => [label.name, label]));
    const normalizedRemoteLabels = remoteLabels
        .filter((label) => /^(?:area|module)\//.test(label.name))
        .map((label) => ({
            name: label.name,
            color: label.color.toUpperCase(),
            description: label.description ?? '',
            ...(expectedByName.get(label.name)?.legacy ? { legacy: true } : {})
        }))
        .sort((left, right) => left.name.localeCompare(right.name));
    const normalizedExpectedLabels = [...labels].sort((left, right) =>
        left.name.localeCompare(right.name)
    );

    assert.deepEqual(
        normalizedRemoteLabels,
        normalizedExpectedLabels,
        `GitHub managed labels differ from ${manifestUrl.pathname}`
    );
}

console.log(
    `Validated ${labels.length} managed labels against .github/mergify.yml` +
        (remoteIndex === -1 ? '' : ' and GitHub')
);
