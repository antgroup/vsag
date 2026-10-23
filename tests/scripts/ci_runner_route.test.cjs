const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const workflow = fs.readFileSync(path.join(root, '.github/workflows/pr-ci.yml'), 'utf8');
const code = workflow.split('// BEGIN ROUTE POLICY\n')[1].split('            // END ROUTE POLICY')[0]
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const route = new (Object.getPrototypeOf(async function(){}).constructor)(
  'github', 'context', 'core', 'Buffer', code);
const image = 'ghcr.io/antgroup/vsag@sha256:' + 'a'.repeat(64);

async function run(options = {}) {
  const outputs = {}, calls = [], warnings = [];
  const eventPR = {number: 7, head: {sha: 'head-a'}, base: {sha: 'base-a'}};
  const context = {repo: {owner: 'antgroup', repo: 'vsag'}, eventName: 'pull_request',
    payload: {pull_request: eventPR}, actor: 'someone-else', ...options.context};
  const pr = {state: 'open', user: {login: 'wxyucs'}, head: {sha: 'head-a', repo: {full_name: 'wxyucs/vsag'}},
    base: {sha: 'base-a', repo: {full_name: 'antgroup/vsag'}}, labels: [], ...options.pr};
  const github = {rest: {
    pulls: {get: async args => {calls.push(['pr', args]); if(options.fail === 'pr') throw Error(); return {data: pr};}},
    repos: {
      getCommit: async args => {calls.push(['main', args]); if(options.fail === 'main') throw Error(); return {data: {sha: 'trusted-main-sha'}};},
      getContent: async args => {
        if (args.path === '.github/ci-routing.json') {
          calls.push(['config', args]);
          if (options.fail === 'config') throw (options.error ?? Error());
          return {data: {type: 'file', encoding: 'base64', size: 100,
            content: Buffer.from(options.configRaw ?? JSON.stringify(options.config ??
              {team_enabled: true, team_image: image})).toString('base64'), ...options.configFile}};
        }
        assert.equal(args.path, '.github/maintainers.json');
        calls.push(['file', args]); if(options.fail === 'file') throw (options.error ?? Error());
        return {data: {type: 'file', encoding: 'base64', size: 100,
          content: Buffer.from(options.raw ?? JSON.stringify(options.policy ?? {maintainers: ['wxyucs']})).toString('base64'),
          ...options.file}};}
    }
  }};
  await route(github, context, {setOutput: (k,v) => outputs[k]=v, info:()=>{}, warning:message=>warnings.push(message)},
    Buffer);
  return {outputs, calls, warnings};
}

test('allowlisted personal-fork PR selects team; actor does not grant access', async () => {
  const {outputs, calls} = await run();
  assert.equal(outputs.team, 'true'); assert.equal(outputs.image, image);
  assert.deepEqual(calls.find(c => c[0] === 'main')[1], {owner:'antgroup',repo:'vsag',ref:'main'});
  assert.equal(calls.filter(c => c[0] === 'main').length, 1);
  assert.deepEqual(calls.find(c => c[0] === 'config')[1],
    {owner:'antgroup',repo:'vsag',path:'.github/ci-routing.json',ref:'trusted-main-sha'});
  assert.deepEqual(calls.find(c => c[0] === 'file')[1],
    {owner:'antgroup',repo:'vsag',path:'.github/maintainers.json',ref:'trusted-main-sha'});
});
for (const [name, options] of Object.entries({
  'disabled': {config: {team_enabled: false, team_image: image}},
  'string enable flag': {config: {team_enabled: 'true', team_image: image}},
  'missing enable flag': {config: {team_image: image}},
  'missing image': {config: {team_enabled: true}},
  'unconfigured image': {config: {team_enabled: true, team_image: ''}},
  'mutable image': {config: {team_enabled: true, team_image: 'ghcr.io/antgroup/vsag:latest'}},
  'other registry': {config: {team_enabled: true, team_image: 'evil.example/vsag@sha256:'+'a'.repeat(64)}},
  'unknown config key': {config: {team_enabled: true, team_image: image, unexpected: true}},
  'missing config': {fail: 'config'},
  'invalid config JSON': {configRaw: '{'},
  'null config': {configRaw: 'null'},
  'array config': {configRaw: '[]'},
  'oversized config': {configFile: {size: 16385}},
  'invalid config size': {configFile: {size: -1}},
  'symlink config': {configFile: {type: 'symlink'}},
  'unsupported config encoding': {configFile: {encoding: 'none'}},
  'manual event': {context: {eventName:'workflow_dispatch'}},
  'noncanonical repository': {context: {repo:{owner:'wxyucs',repo:'vsag'}}},
  'missing PR': {context: {payload:{}}},
  'unlisted author with trusted rerunner': {pr:{user:{login:'outsider'}},context:{actor:'wxyucs'}},
  'closed PR': {pr:{state:'closed'}},
  'new head': {pr:{head:{sha:'head-b'}}},
  'new base': {pr:{base:{sha:'base-b',repo:{full_name:'antgroup/vsag'}}}},
  'other target': {pr:{base:{sha:'base-a',repo:{full_name:'wxyucs/vsag'}}}},
  'missing allowlist': {fail:'file'},
  'main API failure': {fail:'main'},
  'PR API failure': {fail:'pr'},
  'invalid JSON': {raw:'not JSON'},
  'null JSON': {raw:'null'},
  'invalid schema': {policy:{maintainers:'wxyucs'}},
  'malformed login': {policy:{maintainers:['wxyucs','a\nb']}},
  'invalid entry type': {policy:{maintainers:['wxyucs',3]}},
  'empty allowlist': {policy:{maintainers:[]}},
  'oversized policy': {file:{size:16385}},
  'symlink policy': {file:{type:'symlink'}},
})) {
  test(name+' stays on community', async () => {
    const {outputs} = await run(options);
    assert.equal(outputs.team, 'false'); assert.equal(outputs.image, '');
  });
}
test('GitHub usernames are case insensitive', async () => {
  assert.equal((await run({pr:{user:{login:'WXYUCS'}}})).outputs.team, 'true');
});
test('routing never executes checkout or PR-provided code', () => {
  const job = workflow.match(/^  route:\n([\s\S]*?)(?=^  [a-zA-Z_][\w-]*:\n|(?![\s\S]))/m)[1];
  assert.ok(!job.includes('actions/checkout')); assert.ok(!code.includes('require('));
  assert.ok(!code.includes('eval(')); assert.ok(!workflow.includes('pull_request_target'));
});
test('team build uses 24 workers with caller-level FIFO concurrency', () => {
  const team = fs.readFileSync(path.join(root,'.github/workflows/pr-ci-team.yml'),'utf8');
  assert.match(team, /COMPILE_JOBS: '24'/);
  assert.match(team, /make asan DEBUG_BUILD_DIR="\$TEAM_ARTIFACTS\/build" COMPILE_JOBS="\$COMPILE_JOBS"/);
  assert.ok(!/^\s*concurrency:/m.test(team));
  const caller = workflow.split('  team:\n')[1].split('  gate:\n')[0];
  assert.match(caller, /queue: max/);
  assert.match(caller, /cancel-in-progress: false/);
  assert.ok(!team.includes('sudo ')); assert.ok(!team.includes('secrets: inherit'));
});

test('community skips only the migrated Linux checks', () => {
  const text = fs.readFileSync(path.join(root,'.github/workflows/pr-ci-community.yml'),'utf8');
  const jobs = [...text.matchAll(/^  ([a-z][a-z0-9-]*):\n([\s\S]*?)(?=^  [a-z][a-z0-9-]*:\n|$(?![\s\S]))/gm)];
  const skipped = jobs.filter(m => m[2].includes('!inputs.team-linux')).map(m => m[1]);
  assert.deepEqual(skipped, ['build-asan-x86','test-asan-x86-unittests',
    'test-asan-x86-functests','test-example-x86','compatibility']);
});

const {spawnSync} = require('node:child_process');
const gate = workflow.split('      - name: Require the selected CI path to pass\n')[1]
  .split('        run: |\n')[1].split('\n').map(l => l.replace(/^ {10}/,'')).join('\n');
for (const [name, env, success] of [
  ['community success', {}, true],
  ['team success', {TEAM_SELECTED:'true',TEAM_RESULT:'success'}, true],
  ['team failure', {TEAM_SELECTED:'true',TEAM_RESULT:'failure'}, false],
  ['team skipped unexpectedly', {TEAM_SELECTED:'true'}, false],
  ['community failure', {COMMUNITY_RESULT:'failure'}, false],
  ['community cancelled', {COMMUNITY_RESULT:'cancelled'}, false],
  ['policy test failure', {POLICY_TEST_RESULT:'failure'}, false],
  ['route failure', {ROUTE_RESULT:'failure'}, false],
]) {
  test('gate: '+name, () => {
    const result = spawnSync('bash',['-e','-c',gate], {env:{...process.env,
      ROUTE_RESULT:'success',POLICY_TEST_RESULT:'success',COMMUNITY_RESULT:'success',
      TEAM_SELECTED:'false',TEAM_RESULT:'skipped',...env}});
    assert.equal(result.status===0,success);
    if (!success) {
      assert.match(result.stdout.toString(), /::error::.+: expected (success|skipped), got (failure|cancelled|skipped)/);
    }
  });
}


test('routing errors expose stage and status without arbitrary error contents', async () => {
  const error = Object.assign(new Error('secret-url?token=private'), {status: 403});
  const {warnings} = await run({fail: 'file', error});
  assert.match(warnings[0], /read allowlist.*HTTP 403/);
  assert.ok(!warnings[0].includes('private'));
  assert.match((await run({raw: '{'})).warnings[0], /parse allowlist: invalid JSON/);
});

test('team format check has its own clean checkout and remains required', () => {
  const team = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-team.yml'), 'utf8');
  const checks = team.split('  checks:\n')[1].split('  build:\n')[0];
  const summary = team.split('  summary:\n')[1];
  assert.match(checks, /runs-on: ubuntu-22.04/);
  assert.match(checks, /uses: actions\/checkout@v4/);
  assert.match(checks, /persist-credentials: false/);
  assert.ok(checks.includes('./scripts/format/check_format.sh'));
  assert.ok(!checks.includes('mkdir -p ci-logs'));
  assert.match(summary, /needs: \[changes, checks, build, tests, compatibility\]/);
  assert.match(summary, /test "\$CHECK_RESULT" = success/);
});

test('team shard failures remain fatal and diagnostics always upload', () => {
  const team = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-team.yml'), 'utf8');
  assert.ok(!team.includes('continue-on-error'));
  assert.match(team, /fail-fast: false/);
  assert.match(team, /shard: \[0, 1, 2, 3\]/);
  assert.match(team, /--workers 4 --batch-size 32 --batch-target-seconds 25/);
  assert.match(team, /name: Upload test diagnostics\n        if: always\(\)/);
  assert.match(team, /test "\$TEST_RESULT" = success/);
});


test('community CLI download verifies the pinned checksum before extraction', () => {
  const community = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-community.yml'), 'utf8');
  const install = community.split('      - name: Install GitHub CLI\n')[1].split('      - name:')[0];
  const digest = install.match(/GH_SHA256: ([a-f0-9]{64})/)[1];
  const dockerfile = fs.readFileSync(path.join(root, 'docker/Dockerfile.x86.base'), 'utf8');
  assert.ok(dockerfile.includes(`gh_sha256=${digest}`));
  const script = install.split('        run: |\n')[1].split('\n').map(line => line.replace(/^ {10}/, '')).join('\n');
  assert.ok(script.indexOf('sha256sum -c -') < script.indexOf('tar xzf'));
  const temp = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'ci-gh-integrity-'));
  try {
    // Download/command shims avoid network access, root writes, or an installed gh.
    // A corrupt payload must stop the actual workflow shell script before tar runs.
    const shims = `
command() { if [ "$1" = -v ] && [ "$2" = gh ]; then return 1; fi; builtin command "$@"; }
wget() { printf 'corrupt archive' > "$TEST_ARCHIVE"; }
sha256sum() { shasum -a 256 "$@"; }
tar() { touch "$TEST_EXTRACTED"; }
`;
    const result = spawnSync('bash', ['-e', '-o', 'pipefail', '-c', shims + script.replaceAll('/tmp/gh.tar.gz', '"$TEST_ARCHIVE"')], {
      env: {...process.env, GH_VERSION:'2.75.0', GH_SHA256:digest,
        TEST_ARCHIVE:path.join(temp,'gh.tar.gz'), TEST_EXTRACTED:path.join(temp,'extracted')}
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stdout.toString() + result.stderr.toString(), /FAILED/);
    assert.ok(!fs.existsSync(path.join(temp,'extracted')));
  } finally { fs.rmSync(temp, {recursive:true, force:true}); }
});


test('community rejects a mismatched preinstalled CLI before download or use', () => {
  const community = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-community.yml'), 'utf8');
  const install = community.split('      - name: Install GitHub CLI\n')[1].split('      - name:')[0];
  const version = install.match(/GH_VERSION: ([0-9.]+)/)[1];
  const script = install.split('        run: |\n')[1].split('\n')
    .map(line => line.replace(/^ {10}/, '')).join('\n');
  // The container's default shell is sh -e. Test the real step, including its
  // preinstalled fast path, without network calls or writes to system paths.
  const shims = [
    'gh() { printf "gh version %s (fixture)\\nhttps://example.invalid/releases\\n" "$TEST_GH_VERSION"; return "$TEST_GH_EXIT"; }',
    'wget() { echo UNEXPECTED_DOWNLOAD >&2; exit 99; }',
  ].join('\n') + '\n';
  for (const fixture of [
    {version, exit: 0, expected: 0},
    {version: '2.74.0', exit: 0, expected: 1},
    {version, exit: 23, expected: 23},
  ]) {
    const result = spawnSync('sh', ['-e', '-c', shims + script], {
      env: {...process.env, GH_VERSION: version,
        TEST_GH_VERSION: fixture.version, TEST_GH_EXIT: String(fixture.exit)},
      encoding: 'utf8',
    });
    assert.equal(result.status, fixture.expected, JSON.stringify(fixture) + result.stderr);
    assert.ok(!result.stderr.includes('UNEXPECTED_DOWNLOAD'));
  }
});

test('CLI versions agree between image and both workflow paths', () => {
  const dockerfile = fs.readFileSync(path.join(root, 'docker/Dockerfile.x86.base'), 'utf8');
  const team = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-team.yml'), 'utf8');
  const community = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-community.yml'), 'utf8');
  const version = dockerfile.match(/gh_version=([0-9.]+);/)[1];
  const check = team.split('\n').find(line => line.includes('test "$(gh --version'));
  assert.equal(check.match(/= ([0-9.]+)$/)[1], version);
  assert.equal(community.match(/GH_VERSION: ([0-9.]+)/)[1], version);
});

test('compatibility uses verified inputs separate from example output', () => {
  const team = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-team.yml'), 'utf8');
  const compatibility = team.split('      - name: Check index compatibility\n')[1].split('      - name: Record compatibility outcome\n')[0];
  assert.ok(!compatibility.includes('matrix.shard'));
  const compatibilityJob = team.split('  compatibility:\n')[1].split('  summary:\n')[0];
  assert.match(compatibilityJob, /needs: build/);
  assert.match(compatibilityJob, /name: Team Index Compatibility/);
  assert.match(compatibilityJob, /run: \*team_consumer_bootstrap/);
  assert.match(compatibilityJob, /run: \*team_consumer_runtime/);
  assert.match(compatibility, /verify_compatibility_cache.py/);
  assert.match(compatibility, /COMPATIBILITY_INDEX_DIR=\/__ci\/compatibility/);
  assert.ok(compatibility.indexOf('verify_compatibility_cache.py') < compatibility.indexOf('bash ./scripts/check_compatibility.sh'));
  assert.match(team, /--examples-directory examples\/cpp/);
  const mixed = team.split('      - name: Run Catch2 shard and assigned examples\n')[1]
    .split('      - name: Record additional check outcomes\n')[0];
  assert.match(mixed, /id: catch2/);
  assert.match(mixed, /!cancelled\(\) && steps.artifacts.outcome == 'success'/);
  assert.match(mixed, /--plan "\$TEAM_ARTIFACTS\/meta\/plan.json"/);
  assert.match(mixed, /--shard "\$SHARD_INDEX"/);
  assert.match(mixed, /--example-timeout 1800/);
  assert.ok(!mixed.includes('matrix.shard =='));
  assert.match(team, /EXAMPLES_OUTCOME: \$\{\{ steps.catch2.outcome \}\}/);
  const summary = team.split('  summary:\n')[1];
  assert.match(summary, /for index in range\(4\):/);
  assert.match(summary, /needs: \[changes, checks, build, tests, compatibility\]/);
  assert.match(summary, /assert type\(compatibility.get\('schema'\)\) is int and compatibility == expected_compatibility/);
  assert.match(summary, /'context': expected_context/);
  assert.match(summary, /COMPATIBILITY_RESULT: \$\{\{ needs.compatibility.result \}\}/);
  assert.match(summary, /test "\$COMPATIBILITY_RESULT" = success/);
  assert.match(summary, /'examples': 'success'/);
  assert.match(summary, /assert extras == expected/);
  assert.match(summary, /module.merge_results\(plan, results\)/);
  assert.match(summary, /assert summary\['success'\]/);
  assert.ok(!team.includes('      - name: Run examples\n'));
});


test('labels cannot change routing and no repository variables are required', async () => {
  assert.equal((await run({pr: {labels: [{name: 'ci/community'}]}})).outputs.team, 'true');
  assert.ok(!code.includes('process.env'));
  assert.ok(!workflow.includes('vars.CI_TEAM'));
});

test('routing config errors are sanitized and disabled config skips the allowlist', async () => {
  const error = Object.assign(new Error('private credential'), {status: 404});
  const {warnings} = await run({fail: 'config', error});
  assert.match(warnings[0], /read routing config.*HTTP 404/);
  assert.ok(!warnings[0].includes('credential'));
  assert.match((await run({configRaw: '{'})).warnings[0], /parse routing config: invalid JSON/);
  const {calls} = await run({config: {team_enabled: false, team_image: image}});
  assert.ok(!calls.some(call => call[0] === 'file'));
});


test('non-root jobs prepare writable homes before checkout or source verification', () => {
  const teamWorkflow = fs.readFileSync(path.join(root, '.github/workflows/pr-ci-team.yml'), 'utf8');
  for (const name of ['build', 'tests']) {
    const job = teamWorkflow.split(`  ${name}:\n`)[1].split(/\n  [a-z_-]+:\n/)[0];
    assert.ok(job.includes('HOME: /tmp/ci-home'));
    const operation = name === 'build' ? 'uses: actions/checkout' : 'name: Verify read-only artifacts';
    assert.ok(job.indexOf('mkdir -p "$HOME"') < job.indexOf(operation));
  }
});
