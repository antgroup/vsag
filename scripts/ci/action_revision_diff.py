#!/usr/bin/env python3
"""Recognize revision-only changes to external step Actions, never build inputs."""
import re

ACTION = re.compile(r'([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)@'
                    r'(?:v[0-9]+(?:\.[0-9]+)*(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?|[0-9a-fA-F]{40})')


def action_revision_only(before, after):
    # Missing parser or ambiguous YAML means keep the expensive check selected.
    # Do not regex-rewrite YAML: a `uses:` line inside a run block is executable
    # text, not an Action reference. BaseLoader also preserves `on` as a string.
    try:
        import yaml

        class UniqueLoader(yaml.BaseLoader):
            def construct_mapping(self, node, deep=False):
                result = {}
                for key, value in node.value:
                    key = self.construct_object(key, deep=deep)
                    if not isinstance(key, str) or key in result or key == '<<':
                        raise ValueError('ambiguous YAML mapping')
                    result[key] = self.construct_object(value, deep=deep)
                return result

        def normalize(text):
            if len(text) > 1024 * 1024:
                raise ValueError('oversized workflow')
            if any(isinstance(t, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken,
                                  yaml.tokens.TagToken)) for t in yaml.scan(text)):
                raise ValueError('aliases and tags require conservative selection')
            document = yaml.load(text, Loader=UniqueLoader)
            if not isinstance(document, dict):
                raise ValueError('not a workflow or composite Action')
            if 'jobs' in document and isinstance(document['jobs'], dict):
                jobs = list(document['jobs'].values())
            elif isinstance(document.get('runs'), dict) and document['runs'].get('using') == 'composite':
                jobs = [document['runs']]
            else:
                raise ValueError('unknown document shape')
            revisions = []
            for job in jobs:
                if not isinstance(job, dict):
                    raise ValueError('invalid job')
                steps = job.get('steps', [])
                if not isinstance(steps, list):
                    raise ValueError('invalid steps')
                for step in steps:
                    if not isinstance(step, dict):
                        raise ValueError('invalid step')
                    ref = step.get('uses')
                    match = ACTION.fullmatch(ref) if isinstance(ref, str) else None
                    if match:
                        revisions.append(ref)
                        step['uses'] = match[1] + '@<revision>'
            return document, revisions

        old, old_refs = normalize(before)
        new, new_refs = normalize(after)
        return old == new and old_refs != new_refs
    except (ImportError, ValueError, TypeError, RecursionError):
        return False
    except Exception as error:
        # Parser failures are not permission to skip coverage.
        if error.__class__.__module__.startswith('yaml'):
            return False
        raise
