"use strict";
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {resolve} = require("node:path");
const test = require("node:test");

const workflow = readFileSync(resolve(__dirname, "../../.github/workflows/pr-ci-team.yml"), "utf8");
const code = workflow.split("// BEGIN SLOT RECEIPT\n")[1].split("// END SLOT RECEIPT")[0];
const {readAllocation: read, readReviewRejection: readRejection} =
  new Function(code + "\nreturn {readAllocation, readReviewRejection};")();
const expected = {
  appId: "82", repository: "antgroup/vsag", repositoryId: "11", pullRequest: "7",
  runId: "13", attempt: "2", headSha: "a".repeat(40), testSha: "b".repeat(40),
};
const {appId, ...identity} = expected;
const value = {
  schemaVersion: 1, ...identity, slot: "b", runner: "vsag-linux-b", allocationId: "c".repeat(32),
  observedAtMs: 100, expiresAtMs: 200,
};
const check = {
  name: "Team CI slot allocation", app: {id: 82}, external_id: "pr/11/13/2",
  head_sha: expected.testSha, status: "completed", conclusion: "neutral",
  output: {text: JSON.stringify(value)},
};

test("only the approved issuer and exact attempt can select a runner", () => {
  assert.equal(read([check], expected, 151).runner, "vsag-linux-b");
  assert.equal(read([check], {...expected, appId: "9"}, 151), null);
  assert.equal(read([check], {...expected, attempt: "3"}, 151), null);
  assert.equal(read([check], expected, 200), null);
  assert.throws(() => read([check], {...expected, headSha: "d".repeat(40)}, 151));
  assert.throws(() => read([check, check], expected, 151), /Duplicate/);
});

const rejection = {
  ...check, external_id: "review/pr/11/13/2",
  output: {text: JSON.stringify({schemaVersion: 1, ...identity, reason: "workflow-review-required"})},
};

test("a review rejection belongs to the configured App and exact attempt", () => {
  assert.equal(readRejection([rejection], expected), true);
  assert.equal(readRejection([{...rejection, app: {id: 9}}], expected), false);
  assert.equal(readRejection([rejection], {...expected, attempt: "3"}), false);
  assert.equal(readRejection([check], expected), false);
  assert.equal(read([rejection], expected, 151), null);
  assert.throws(() => readRejection([rejection], {...expected, testSha: "d".repeat(40)}), /identity/);
  assert.throws(() => readRejection([rejection], {...expected, headSha: "d".repeat(40)}), /attempt/);
  assert.throws(() => readRejection([rejection, rejection], expected), /Duplicate/);
});

async function runWaiter(paginate) {
  const slot = workflow.split("\n  slot:\n")[1].split("\n  build:\n")[0];
  const script = slot.split("\n          script: |\n")[1];
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const run = new AsyncFunction("github", "context", "core", "process", "Date", "setTimeout", script);
  const failures = [];
  const outputs = {};
  let now = 151;
  await run(
    {paginate, rest: {checks: {listForRef() {}}}},
    {
      eventName: "pull_request",
      repo: {owner: "antgroup", repo: "vsag"},
      payload: {repository: {id: 11}, pull_request: {number: 7, base: {ref: "main"}}},
    },
    {info() {}, setOutput(key, value) { outputs[key] = value; }, setFailed(message) { failures.push(message); }},
    {env: {
      SLOT_APP_ID: appId, GITHUB_RUN_ID: expected.runId, GITHUB_RUN_ATTEMPT: expected.attempt,
      TEAM_HEAD_SHA: expected.headSha, TEAM_TEST_SHA: expected.testSha,
    }},
    {now: () => now},
    (callback, milliseconds) => { now += milliseconds; callback(); },
  );
  return {failures, outputs};
}

test("the hosted waiter retries a temporary API error and names the workflow reviewer on rejection", async () => {
  let polls = 0;
  const result = await runWaiter(async () => {
    if (++polls === 1) throw Object.assign(new Error("unavailable"), {status: 503});
    return [rejection];
  });
  assert.equal(polls, 2);
  assert.deepEqual(result, {
    failures: ["Team CI blocked because the workflow digest has not been approved. @wxyucs must review the workflow changes and append the trusted digest for this PR, then select Re-run all jobs."],
    outputs: {},
  });
});

test("the hosted waiter identifies persistent Checks API errors", async () => {
  await assert.rejects(
    runWaiter(async () => { throw Object.assign(new Error("unavailable"), {status: 503}); }),
    /GitHub Checks API was unavailable \(HTTP 503\)/,
  );
});

test("the hosted waiter still accepts a normal slot allocation", async () => {
  assert.deepEqual(await runWaiter(async () => [check]), {
    failures: [], outputs: {runner: "vsag-linux-b"},
  });
});

test("the waiter runs hosted and all six internal jobs share its output", () => {
  const slot = workflow.split("\n  slot:\n")[1].split("\n  build:\n")[0];
  assert.match(slot, /runs-on: ubuntu-24\.04/);
  assert.match(slot, /checks: read/);
  assert.doesNotMatch(slot, /actions\/checkout/);
  const selectedRunner = "runs-on: ${{ inputs.slot-app-id != '' && needs.slot.outputs.runner || inputs.runner }}";
  assert.equal(workflow.split(selectedRunner).length - 1, 3);
  assert.match(workflow, /shard: \[0, 1, 2, 3\]/);
  const build = workflow.split("\n  build:\n")[1].split("\n  tests:\n")[0];
  assert.match(build, /inputs\.slot-app-id == '' && needs\.slot\.result == 'skipped'/);
  assert.match(build, /needs\.slot\.result == 'success' && needs\.slot\.outputs\.runner != ''/);
});
