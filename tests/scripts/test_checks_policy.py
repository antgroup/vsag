#!/usr/bin/env python3
"""Event/path routing regressions for the consolidated PR and branch checks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('checks_policy', ROOT / 'scripts/ci/checks_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class Selection(unittest.TestCase):
    def pr(self, paths, branch='main', action='opened'):
        return policy.select('pull_request', {'action': action, 'pull_request': {'base': {'ref': branch}}}, paths)

    def test_pr_scope_matrix(self):
        for branch, paths, expected in [
            ('main', ['README.md'], {'issue'}),
            ('main', ['docs/guide.md'], {'docs', 'issue'}),
            ('main', ['src/index.cpp'], {'docs', 'issue'}),
            ('main', ['src/simd/kernel.cpp'], {'docs', 'issue', 'simd', 'lite'}),
            ('0.18', ['src/simd/kernel.cpp'], {'docs', 'issue', 'lite'}),
            ('feature/other', ['src/simd/kernel.cpp'], {'docs'}),
            ('main', ['Makefile'], {'issue', 'simd'}),
            ('main', ['.clang-format'], {'issue', 'lite'}),
            ('0.x/nested', ['include/vsag/lite/a.h'], {'docs'}),
        ]:
            with self.subTest(branch=branch, paths=paths):
                self.assertEqual({k for k,v in self.pr(paths, branch).items() if v}, expected)

    def test_metadata_events_only_check_issue(self):
        for action in ('edited', 'labeled', 'unlabeled'):
            self.assertEqual({k for k,v in self.pr(['src/simd/a.cpp'], action=action).items() if v}, {'issue'})
            self.assertFalse(any(self.pr(['docs/a.md'], 'feature', action).values()))

    def test_default_actions_and_ignored_actions(self):
        for action in ('opened', 'synchronize', 'reopened'):
            self.assertTrue(self.pr(['src/simd/a.cpp'], action=action)['simd'])
        self.assertFalse(any(self.pr(['src/simd/a.cpp'], action='closed').values()))

    def test_push_scope(self):
        for branch, paths, expected in [
            ('main', ['docs/a.md'], {'docs'}),
            ('main', ['README.md'], {'format'}),
            ('main', ['src/a.cpp'], {'format','docs'}),
            ('0.18', ['docs/a.md'], set()),
            ('0.18', ['src/a.cpp'], {'format'}),
            ('feature', ['src/a.cpp'], set()),
            ('main', [], set()),
        ]:
            result=policy.select('push', {'ref':'refs/heads/'+branch}, paths)
            self.assertEqual({k for k,v in result.items() if v}, expected)

    def test_manual_checks_do_not_expand_scope(self):
        for choice in ('format','docs','lite'):
            result=policy.select('workflow_dispatch', {'inputs':{'check':choice}}, [])
            self.assertEqual({k for k,v in result.items() if v}, {choice})
        with self.assertRaises(ValueError):
            policy.select('workflow_dispatch', {'inputs':{'check':'simd'}}, [])

    def test_workflow_wiring(self):
        pr = (ROOT / '.github/workflows/pr-ci.yml').read_text()
        branch = (ROOT / '.github/workflows/main-ci.yml').read_text()
        self.assertNotIn('  push:', pr)
        self.assertNotIn('  pull_request:', branch)
        self.assertIn("branches: [main, '0.*']", branch)
        self.assertIn('  workflow_call:', pr)
        self.assertNotIn('  workflow_dispatch:', pr)
        self.assertIn('  workflow_call:', branch)
        self.assertNotIn('  workflow_dispatch:', branch)
        community = (ROOT / '.github/workflows/pr-ci-community.yml').read_text()
        for key in ('docs','simd','lite'):
            self.assertIn(f'if: inputs.run-{key}', community)
            self.assertIn(f"run-{key}: ${{{{ needs.select-checks.outputs.{key} == 'true' }}}}", pr)
        for text, keys in ((pr, ()), (branch, ('docs',))):
            for key in keys:
                self.assertIn(f"needs.select-checks.outputs.{key} == 'true'", text)
            self.assertIn('fetch-depth: 0', text)
            self.assertIn('persist-credentials: false', text)
            self.assertNotIn('pull_request_target', text)
        self.assertIn('\nconcurrency:\n', pr)
        self.assertEqual(pr.count('    concurrency:'), 1)
        self.assertNotIn('\nconcurrency:', community)
        self.assertNotIn('    concurrency:', community.split('  version-tsan-changes:', 1)[0])
        self.assertIn("github.event.deleted != true", branch)

    def test_deleted_branch_never_diffs_or_selects_checks(self):
        event = {'ref':'refs/heads/0.18', 'deleted':True, 'after':'0'*40}
        with patch.object(policy, 'git', side_effect=AssertionError('must not invoke Git')):
            self.assertEqual(policy.changed_paths('push', event), [])
        self.assertFalse(any(policy.select('push', event, ['src/simd/a.cpp']).values()))

    def test_metadata_and_manual_entries_are_separate(self):
        import re
        def workflow(name):
            text = (ROOT / '.github/workflows' / name).read_text()
            # Inspect only event declarations and top-level job fields; runtime
            # GitHub expression tests below cover eligibility and failure guards.
            triggers = text.split('on:\n',1)[1].split('\npermissions:',1)[0].split('\nconcurrency:',1)[0]
            events = {key: {} for key in re.findall(r'^  ([a-z_]+):', triggers, re.M)}
            if 'pull_request' in events:
                types = re.search(r'types: \[(.*?)\]', triggers).group(1)
                events['pull_request']['types'] = [value.strip() for value in types.split(',')]
            jobs = {}
            for key, body in re.findall(r'^  ([\w-]+):\n(.*?)(?=^  [\w-]+:\n|\Z)', text.split('jobs:\n',1)[1], re.M|re.S):
                jobs[key] = dict(re.findall(r'^    (name|if|uses): (.*)$', body, re.M))
            return {'true':events, 'jobs':jobs}
        pr = workflow('pr-ci.yml')
        metadata = workflow('pr-issue-link.yml')
        manual = workflow('manual-checks.yml')
        branch = workflow('main-ci.yml')
        self.assertEqual(set(pr['true']['pull_request']['types']), {'opened','reopened','synchronize'})
        self.assertEqual(set(metadata['true']['pull_request']['types']),
                         {'opened','edited','reopened','synchronize','labeled','unlabeled'})
        self.assertEqual(set(metadata['jobs']), {'check-issue-link'})
        self.assertEqual(set(manual['true']), {'workflow_dispatch'})
        for document in (pr, metadata, branch):
            self.assertNotIn('workflow_dispatch', document['true'])
            for item in document['jobs'].values():
                name = item['name']
                self.assertNotIn('&&', name)
                self.assertNotIn('||', name)
                self.assertNotIn('github.', name)
        for item in manual['jobs'].values():
            self.assertTrue(item['name'].startswith('Manual '))
        self.assertEqual(manual['jobs']['pr']['if'], "inputs.check == 'full-pr'")
        self.assertIn('full-branch', manual['jobs']['branch']['if'])
        self.assertIn('inputs.check', manual['jobs']['branch']['if'])
        self.assertEqual(pr['jobs']['gate']['name'], '${{ matrix.check }}')
        self.assertEqual(branch['jobs']['clang-format-check']['name'], '${{ matrix.check }}')
        for choice in ('full-pr','full-branch'):
            self.assertFalse(any(policy.select('workflow_dispatch', {'inputs': {'check': choice}}, []).values()))

    def test_manual_bodies_match_automatic_checks(self):
        import re
        def body(file, key):
            text = (ROOT / '.github/workflows' / file).read_text()
            return re.search(r'^  '+key+r':\n.*?(?=^  [\w-]+:\n|\Z)', text, re.M|re.S).group()
        for key in ('validate','build-test-install'):
            automatic = body('pr-ci-community.yml', key).split('    runs-on:',1)[1]
            manual = body('manual-checks.yml', key).split('    runs-on:',1)[1]
            self.assertEqual(automatic.strip(), manual.strip())
        automatic = body('main-ci.yml','clang-format-check').split('      - name: Checkout code',1)[1]
        manual = body('manual-checks.yml','clang-format-check').split('      - name: Checkout code',1)[1]
        automatic = re.sub(r'^        if: matrix.check.*\n', '', automatic, flags=re.M)
        self.assertEqual(automatic.strip(), manual.strip())

    def test_top_level_and_community_gate(self):
        import re
        import itertools
        import textwrap
        pr = (ROOT / '.github/workflows/pr-ci.yml').read_text()
        self.assertEqual(set(re.findall(r'^  ([\w-]+):', pr.split('jobs:\n',1)[1], re.M)),
                         {'route','routing-tests','select-checks','community','team','gate'})
        gate = pr.split('  gate:\n',1)[1]
        self.assertIn('needs: [route, routing-tests, select-checks, community, team]', gate)
        self.assertIn('if: always()', gate)
        community = (ROOT / '.github/workflows/pr-ci-community.yml').read_text()
        result = re.search(r'(?ms)^  shared-checks-result:\n(.*?)(?=^  [\w-]+:\n|\Z)', community)[1]
        self.assertIn('needs: [validate, simd-coverage, build-test-install]', result)
        self.assertIn('if: always()', result)
        script = textwrap.dedent(result.split('        run: |\n',1)[1])
        # All selections and each failing/cancelled/unexpectedly-skipped check
        # exercise the exact summary shell that determines reusable success.
        for flags in itertools.product((False, True), repeat=3):
            env = dict(os.environ)
            for key, enabled in zip(('DOCS','SIMD','LITE'), flags):
                env['RUN_'+key] = str(enabled).lower()
                env[key+'_RESULT'] = 'success' if enabled else 'skipped'
            self.assertEqual(subprocess.run(['bash','-e','-c',script],env=env,capture_output=True).returncode,0)
            for key in ('DOCS','SIMD','LITE'):
                for bad in ('failure','cancelled', 'skipped' if env['RUN_'+key]=='true' else 'success'):
                    broken = {**env,key+'_RESULT':bad}
                    self.assertNotEqual(subprocess.run(['bash','-e','-c',script],env=broken,capture_output=True).returncode,0)
        # Auxiliary jobs run regardless of team-linux/core. They belong to the
        # hosted Community group without extending the Team concurrency lock.
        for key in ('validate','simd-coverage','build-test-install'):
            job = re.search(r'^  '+key+r':\n.*?(?=^  [\w-]+:\n|\Z)',community,re.M|re.S).group()
            self.assertNotIn('team-linux',job)
            self.assertNotIn('inputs.core',job)

    def test_live_issue_metadata_rejects_stale_head_and_closed_pr(self):
        import json
        workflow = (ROOT / '.github/workflows/pr-issue-link.yml').read_text()
        job = workflow.split('  check-issue-link:', 1)[1].split('  simd-coverage:', 1)[0]
        self.assertNotIn('needs: select-checks', job)
        snippet = job.split('          script: |\n', 1)[1].split('      - name:', 1)[0]
        script = '\n'.join(line[12:] for line in snippet.splitlines())
        harness = r"""
const script = process.argv[1];
(async () => {
 let results = [];
 for (const [head, state] of [['same','open'],['new','open'],['same','closed']]) {
  const outputs = {};
  const github = {rest:{pulls:{get:async()=>({data:{head:{sha:head},state,
    body:'Fixes: #123',labels:[{name:'kind/bug'}]}})}}};
  const context = {repo:{owner:'antgroup',repo:'vsag'},issue:{number:1},
    payload:{pull_request:{head:{sha:'same'},body:'outdated',labels:[]}}};
  const core = {setOutput:(k,v)=>outputs[k]=v};
  try {
   await new (Object.getPrototypeOf(async function(){}).constructor)('github','context','core',script)(github,context,core);
   results.push({ok:true,outputs});
  } catch(e) { results.push({ok:false,outputs}); }
 }
 console.log(JSON.stringify(results));
})();
"""
        results = json.loads(subprocess.check_output(['node','-e',harness,script],text=True))
        self.assertEqual(results[0], {'ok':True,'outputs':{'body':'Fixes: #123','labels':'["kind/bug"]'}})
        self.assertEqual(results[1:], [{'ok':False,'outputs':{}},{'ok':False,'outputs':{}}])

    def test_issue_link_interleaving_uses_shared_head_lock(self):
        import json
        import re
        workflow = (ROOT / '.github/workflows/pr-issue-link.yml').read_text()
        job = workflow.split('  check-issue-link:', 1)[1].split('  simd-coverage:', 1)[0]
        group = re.search(r'^      group: (.*)$', job, re.M).group(1)
        self.assertIn('cancel-in-progress: true', job)
        self.assertNotIn('needs: select-checks', job)
        # The lock is job-scoped: live metadata is read only after acquiring it.
        self.assertLess(job.index('    concurrency:'), job.index('    steps:'))
        self.assertLess(job.index('Read current PR metadata'), job.index('Validate PR description'))
        harness = r"""
const template = process.argv[1];
const keys = ['synchronize','edited','labeled','unlabeled'].map(action => {
 const github = {event:{action,pull_request:{number:42,head:{sha:'current'}}}};
 return template.replace(/\$\{\{ (.*?) \}\}/g, (_, expr) => eval(expr));
});
const github = {event:{action:'synchronize',pull_request:{number:42,head:{sha:'old'}}}};
keys.push(template.replace(/\$\{\{ (.*?) \}\}/g, (_, expr) => eval(expr)));
console.log(JSON.stringify(keys));
"""
        keys = json.loads(subprocess.check_output(['node','-e',harness,group],text=True))
        self.assertEqual(len(set(keys[:4])),1)
        self.assertNotEqual(keys[0],keys[4])
        # Model GitHub cancel-in-progress with the evaluated production keys:
        # old code job has saved a valid body; edited body then fails. The old
        # job is cancelled, so its attempted late success cannot replace failure.
        active = {}
        states = {}
        def start(run, key):
            if key in active:
                states[active[key]] = 'cancelled'
            active[key] = run
            states[run] = 'running'
        start('code', keys[0])
        start('metadata', keys[1])
        states['metadata'] = 'failure'
        if states['code'] == 'running':
            states['code'] = 'success'
        self.assertEqual(states, {'code':'cancelled','metadata':'failure'})
        start('stale-head', keys[4])
        self.assertEqual(states['metadata'],'failure')

    def test_merged_core_event_guards(self):
        import json
        import re
        pr=(ROOT / '.github/workflows/pr-ci.yml').read_text()
        branch=(ROOT / '.github/workflows/main-ci.yml').read_text()
        def expression(text, job):
            block=text.split('  '+job+':\n',1)[1].split('\n  ',1)[0]
            # Read only top-level job fields, not nested step fields.
            block=text.split('  '+job+':\n',1)[1]
            return re.search(r'^    if: (.*)$',block,re.M).group(1).removeprefix('${{').removesuffix('}}').strip()
        expressions={key:expression(pr,key) for key in ('route','routing-tests','team')}
        expressions.update({key:expression(branch,key) for key in ('changes','summary')})
        harness=r"""
const expressions=JSON.parse(process.argv[1]);
const fromJSON=JSON.parse, contains=(a,b)=>a.includes(b), startsWith=(a,b)=>a.startsWith(b), always=()=>true;
const cases=[
 ['pull_request','opened','main','',false], ['pull_request','synchronize','0.18','',false],
 ['pull_request','reopened','feature','',false], ['pull_request','edited','main','',false],
 ['pull_request','labeled','main','',false], ['pull_request','unlabeled','main','',false],
 ['workflow_dispatch','','main','full',false], ['workflow_dispatch','','main','docs',false],
 ['workflow_dispatch','','main','lite',false], ['workflow_dispatch','','main','format',false],
 ['push','','main','',false], ['push','','0.18','',false], ['push','','main','',true]
];
const out=cases.map(([event_name,action,base_ref,check,deleted])=>{
 const github={event_name,base_ref,ref:'refs/heads/'+base_ref,event:{action,deleted,changes:{}}};
 const inputs={check,profile:''}; const needs={route:{outputs:{team:'true'}}};
 return Object.fromEntries(Object.entries(expressions).map(([k,e])=>[k,Boolean(eval(e))]));
});
console.log(JSON.stringify(out));
"""
        results=json.loads(subprocess.check_output(['node','-e',harness,json.dumps(expressions)],text=True))
        for i,result in enumerate(results):
            for key in ('route','routing-tests','team'):
                self.assertEqual(result[key],i in (0,1,6,7,8,9),(i,key))
            for key in ('changes','summary'):
                self.assertEqual(result[key],i in (6,7,8,9,10),(i,key))
        self.assertFalse((ROOT / '.github/workflows/pr-checks.yml').exists())
        self.assertFalse((ROOT / '.github/workflows/branch-checks.yml').exists())
        self.assertFalse(any(policy.select('workflow_dispatch',{'inputs':{'check':'full'}},[]).values()))

    def test_required_format_propagates_selection_failure(self):
        import json
        import re
        import textwrap
        workflow=(ROOT / '.github/workflows/main-ci.yml').read_text()
        job=workflow.split('  clang-format-check:\n',1)[1].split('\n  validate:',1)[0]
        expression=re.search(r'^        check: \$\{\{ (.*) \}\}$',job,re.M).group(1)
        self.assertRegex(job, r'    if: (?:\$\{\{ )?always\(\)')
        harness=r"""
const expression=process.argv[1]; const always=()=>true; const fromJSON=JSON.parse;
let out=[];
for (const [result,format,deleted] of [
 ['success','true',false], ['success','false',false],
 ['failure','',false], ['cancelled','',false], ['skipped','',false],
 ['failure','',true], ['skipped','',true]
]) {
 const github={event:{deleted}};
 const needs={'select-checks':{result,outputs:{format}}};
 // GitHub permits hyphens in property identifiers; JavaScript needs brackets.
 out.push(eval(expression.replaceAll('needs.select-checks', "needs['select-checks']"))[0] === "Format Check");
}
console.log(JSON.stringify(out));
"""
        results=json.loads(subprocess.check_output(['node','-e',harness,expression],text=True))
        self.assertEqual(results,[True,False,True,True,True,False,False])
        guard=job.split('      - name: Require successful check selection\n',1)[1].split('      - name: Checkout code',1)[0]
        script=textwrap.dedent(guard.split('        run: |\n',1)[1])
        for result in ('success','failure','cancelled','skipped',''):
            completed=subprocess.run(['bash','-e','-c',script],env={**os.environ,'SELECTION_RESULT':result},capture_output=True,text=True)
            self.assertEqual(completed.returncode==0,result=='success')
            if result!='success':
                self.assertIn('::error::Check selection did not succeed',completed.stdout)

    def test_gate_and_branch_contexts_and_retarget(self):
        import json
        import re
        pr = (ROOT / '.github/workflows/pr-ci.yml').read_text()
        branch = (ROOT / '.github/workflows/main-ci.yml').read_text()
        def job(text,key):
            return re.search(r'^  '+key+r':\n.*?(?=^  [\w-]+:\n|\Z)',text,re.M|re.S).group()
        def matrix(text,key):
            return re.search(r'^        check: \$\{\{ (.*) \}\}$',job(text,key),re.M).group(1)
        expressions = {'gate':matrix(pr,'gate'), 'format':matrix(branch,'clang-format-check'),
                       'docs':matrix(branch,'validate'),
                       'core':re.search(r'^    if: (.*)$',job(pr,'route'),re.M).group(1),
                       'select':'true',
                       'group':re.search(r'^  group: (.*)$',pr,re.M).group(1)}
        harness = r"""
const es=JSON.parse(process.argv[1]);
const fromJSON=JSON.parse,contains=(a,b)=>a.includes(b),startsWith=(a,b)=>a.startsWith(b);
const results=[];
for (const [base_ref,action,changes,deleted,selected] of [
 ['main','opened',{},false,'true'],['feature','opened',{},false,'true'],
 ['main','edited',{base:{ref:{from:'feature'}}},false,'true'],
 ['0.18','edited',{base:{ref:{from:'main'}}},false,'true'],
 ['main','edited',{body:{from:'old'}},false,'true'],
 ['main','opened',{},true,'true'],['0.18','opened',{},false,'true'],
 ['main','opened',{},false,'false']
]) {
 const github={event_name:'pull_request',base_ref,ref:'refs/heads/'+base_ref,event:{action,changes,deleted,pull_request:{number:42}}};
 const needs={'select-checks':{result:'success',outputs:{format:selected,docs:selected}}};
 const evalExpr=e=>eval(e.replaceAll('needs.select-checks',"needs['select-checks']"));
 results.push(Object.fromEntries(Object.entries(es).map(([k,e])=>[k,k==='group'?e.replace(/\$\{\{ (.*?) \}\}/g,(_,x)=>evalExpr(x)):evalExpr(e)])));
}
console.log(JSON.stringify(results));
"""
        rows=json.loads(subprocess.check_output(['node','-e',harness,json.dumps(expressions)],text=True))
        self.assertEqual([x['gate'][0] for x in rows[:5]], ['PR CI Gate','Docs Gate','Docs Gate','Docs Gate','Docs Gate'])
        self.assertEqual([x['core'] for x in rows[:5]], [True,False,False,False,False])
        self.assertNotIn('Metadata Only', pr)
        self.assertNotIn('edited', pr.split('permissions:', 1)[0])
        self.assertEqual(rows[0]['group'],rows[2]['group'])
        self.assertEqual(rows[0]['group'],rows[4]['group'])
        for i in (5,7):
            self.assertEqual(rows[i]['format'],['Format Check Not Required'])
            self.assertEqual(rows[i]['docs'],['Branch Docs Not Required'])
        self.assertEqual(rows[6]['format'],['Format Check'])
        self.assertEqual(rows[6]['docs'],['Branch Docs Not Required'])
        for key in ('clang-format-check','validate'):
            self.assertRegex(job(branch,key), r'    if: (?:\$\{\{ )?always\(\)')
        manual=(ROOT / '.github/workflows/manual-checks.yml').read_text()
        self.assertIn('      actions: read',job(manual,'pr'))
        self.assertNotIn('actions:',manual.split('jobs:',1)[0])
        event={'action':'edited','changes':{'base':{'ref':{'from':'feature'}}},'pull_request':{'base':{'ref':'main'}}}
        self.assertEqual({k for k,v in policy.select('pull_request',event,['src/simd/a.cpp']).items() if v}, {'docs','simd','lite','issue'})

    def test_globs(self):
        self.assertTrue(policy.matches('src/CMakeLists.txt','src/**/CMakeLists.txt'))
        self.assertTrue(policy.matches('src/a/b/CMakeLists.txt','src/**/CMakeLists.txt'))
        self.assertFalse(policy.matches('src/a.cpp','src/simd/**'))

    def test_metadata_and_manual_need_no_diff(self):
        with patch.object(policy, 'git', side_effect=AssertionError('must not invoke Git')):
            self.assertEqual(policy.changed_paths('workflow_dispatch', {}), [])
            self.assertEqual(policy.changed_paths('pull_request', {'action':'edited'}), [])


class GitRanges(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.previous=Path.cwd();os.chdir(self.directory.name)
        self.addCleanup(os.chdir,self.previous)
        self.git('init','-q')
        self.git('config','user.name','Test')
        self.git('config','user.email','test@example.invalid')
        self.commit('README.md')

    def git(self,*args):
        return subprocess.check_output(['git',*args],stderr=subprocess.PIPE).decode().strip()

    def commit(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture\n')
        self.git('add','-A');self.git('commit','-qm','fixture')
        return self.git('rev-parse','HEAD')

    def test_push_full_range_deletion_and_rename(self):
        base=self.git('rev-parse','HEAD')
        self.commit('src/simd/a.cpp')
        self.git('mv','README.md','guide.md')
        self.git('commit','-qm','rename')
        head=self.git('rev-parse','HEAD')
        self.assertEqual(set(policy.changed_paths('push',{'before':base,'after':head})), {'README.md','guide.md','src/simd/a.cpp'})

    def test_pr_uses_three_dot_not_two_dot(self):
        base=self.git('rev-parse','HEAD')
        self.git('checkout','-qb','pr')
        head=self.commit('docs/a.md')
        self.git('checkout','--detach',base)
        advanced=self.commit('src/simd/main-only.cpp')
        event={'action':'opened','pull_request':{'base':{'sha':advanced},'head':{'sha':head}}}
        self.assertEqual(policy.changed_paths('pull_request',event),['docs/a.md'])
        event.update(action='edited', changes={'base': {'ref': {'from': 'feature'}}})
        self.assertEqual(policy.changed_paths('pull_request',event),['docs/a.md'])

    def test_new_branch_uses_all_pushed_commits(self):
        first=self.commit('src/a.cpp');head=self.commit('docs/a.md')
        event={'before':'0'*40,'after':head,'commits':[{'id':first},{'id':head}]}
        self.assertEqual(set(policy.changed_paths('push',event)),{'src/a.cpp','docs/a.md'})

    def test_unknown_range_fails_not_empty_success(self):
        head=self.git('rev-parse','HEAD')
        with self.assertRaises(ValueError):
            policy.changed_paths('push',{'before':'0'*40,'after':head})
        with self.assertRaises(subprocess.CalledProcessError):
            policy.changed_paths('push',{'before':'f'*40,'after':head})
        with self.assertRaises(ValueError):
            policy.changed_paths('push',{'before':'--help','after':head})


if __name__=='__main__':
    unittest.main()
