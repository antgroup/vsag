"use strict";
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {resolve} = require("node:path");
const test = require("node:test");

const workflow = readFileSync(resolve(__dirname, "../../.github/workflows/pr-ci-team.yml"), "utf8");
const code = workflow.split("// BEGIN SLOT RECEIPT\n")[1].split("// END SLOT RECEIPT")[0];
const read = new Function(code + "\nreturn readAllocation;")();
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
