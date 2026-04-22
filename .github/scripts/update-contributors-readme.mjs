import { readFile, writeFile } from 'node:fs/promises';

const CONTRIBUTORS_PER_ROW = 6;
const START_MARKER = '<!-- readme: contributors -start -->';
const END_MARKER = '<!-- readme: contributors -end -->';

function parseArgs(argv) {
    const args = {};

    for (let index = 0; index < argv.length; index += 1) {
        const token = argv[index];
        const next = argv[index + 1];

        if (!token.startsWith('--') || next === undefined) {
            continue;
        }

        args[token.slice(2)] = next;
        index += 1;
    }

    return args;
}

function escapeHtml(value) {
    return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

async function githubRequest(path, token) {
    const response = await fetch(`https://api.github.com${path}`, {
        headers: {
            Accept: 'application/vnd.github+json',
            Authorization: `Bearer ${token}`,
            'User-Agent': 'vsag-contributors-readme',
            'X-GitHub-Api-Version': '2022-11-28'
        }
    });

    if (!response.ok) {
        throw new Error(`GitHub API request failed for ${path}: ${response.status}`);
    }

    return response.json();
}

async function listContributors(owner, repo, token) {
    const contributors = [];

    for (let page = 1; ; page += 1) {
        const pageItems = await githubRequest(
            `/repos/${owner}/${repo}/contributors?per_page=100&page=${page}`,
            token
        );

        contributors.push(...pageItems);

        if (pageItems.length < 100) {
            return contributors;
        }
    }
}

function parseExistingNames(readmeContent) {
    const namesByLogin = new Map();
    const startIndex = readmeContent.indexOf(START_MARKER);
    const endIndex = readmeContent.indexOf(END_MARKER);

    if (startIndex === -1 || endIndex === -1 || endIndex <= startIndex) {
        return namesByLogin;
    }

    const existingBlock = readmeContent.slice(startIndex, endIndex + END_MARKER.length);
    const contributorCellPattern =
        /<a href="https:\/\/github\.com\/([^\"]+)">[\s\S]*?<sub><b>([^<]+)<\/b><\/sub>/g;

    let match = contributorCellPattern.exec(existingBlock);

    while (match !== null) {
        namesByLogin.set(match[1], match[2]);
        match = contributorCellPattern.exec(existingBlock);
    }

    return namesByLogin;
}

async function resolveDisplayName(login, namesByLogin, token) {
    const existingName = namesByLogin.get(login);

    if (existingName) {
        return existingName;
    }

    const user = await githubRequest(`/users/${encodeURIComponent(login)}`, token);
    return user.name || login;
}

function renderContributorCell(contributor) {
    const login = escapeHtml(contributor.login);
    const avatarUrl = escapeHtml(contributor.avatar_url);
    const displayName = escapeHtml(contributor.name);

    return [
        '            <td align="center">',
        `                <a href="https://github.com/${login}">`,
        `                    <img src="${avatarUrl}" width="100;" alt="${login}"/>`,
        '                    <br />',
        `                    <sub><b>${displayName}</b></sub>`,
        '                </a>',
        '            </td>'
    ];
}

function renderContributorsTable(contributors) {
    const lines = [START_MARKER, '<table>', '    <tbody>'];

    for (let index = 0; index < contributors.length; index += CONTRIBUTORS_PER_ROW) {
        lines.push('        <tr>');

        for (const contributor of contributors.slice(index, index + CONTRIBUTORS_PER_ROW)) {
            lines.push(...renderContributorCell(contributor));
        }

        lines.push('        </tr>');
    }

    lines.push('    </tbody>', '</table>', END_MARKER);

    return `${lines.join('\n')}\n`;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
    const repository = args.repo || process.env.GITHUB_REPOSITORY;
    const readmePath = args.readme || 'README.md';

    if (!token) {
        throw new Error('GITHUB_TOKEN or GH_TOKEN is required');
    }

    if (!repository || !repository.includes('/')) {
        throw new Error('Repository must be provided via --repo or GITHUB_REPOSITORY');
    }

    const [owner, repo] = repository.split('/');
    const readmeContent = await readFile(readmePath, 'utf8');
    const startIndex = readmeContent.indexOf(START_MARKER);
    const endIndex = readmeContent.indexOf(END_MARKER);

    if (startIndex === -1 || endIndex === -1 || endIndex <= startIndex) {
        throw new Error(`Could not find contributors markers in ${readmePath}`);
    }

    const existingNames = parseExistingNames(readmeContent);
    const contributors = (await listContributors(owner, repo, token))
        .filter(user => user.type !== 'Bot' && !user.login.includes('actions-user'))
        .sort(
            (left, right) =>
                right.contributions - left.contributions || left.login.localeCompare(right.login)
        );

    const contributorsWithNames = [];

    for (const contributor of contributors) {
        contributorsWithNames.push({
            ...contributor,
            name: await resolveDisplayName(contributor.login, existingNames, token)
        });
    }

    const updatedBlock = renderContributorsTable(contributorsWithNames);
    const updatedReadme =
        readmeContent.slice(0, startIndex) +
        updatedBlock +
        readmeContent.slice(endIndex + END_MARKER.length + 1);

    if (updatedReadme === readmeContent) {
        return;
    }

    await writeFile(readmePath, updatedReadme, 'utf8');
}

main().catch(error => {
    console.error(error.message);
    process.exitCode = 1;
});
