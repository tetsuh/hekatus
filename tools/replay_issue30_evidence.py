"""Replay retained Issue30 bytes in private copies; never issue board authority."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from enodia.tt.bench import issue30 as m

CASES = (("A", "issue30-a-discovery-watcher-v1-789de6a8faf8"),
         ("B", "issue30-b-fabric1d-smoke-watcher-v1-82c9b462d5ba"),
         ("A", "issue30-a-discovery-watcher-v2-c1f49b70"),
         ("B", "issue30-b-fabric1d-smoke-watcher-v2-c1f49b70"))


class Copies:
    def __init__(self, source, snapshots):
        self.source, self.snapshots, self.protected = source, snapshots, {}

    def copy(self, path, target):
        root = self.source if path.is_relative_to(self.source) else self.snapshots
        raw = m._read(path, root, limit=m.diagnostics.checks.MAX_WATCHER_BYTES)
        self.protected[str(path)] = m.digest(raw)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        target.chmod(0o600)

    def verify_originals(self):
        for name, sha in self.protected.items():
            if m.digest(Path(name).read_bytes()) != sha:
                raise RuntimeError('original evidence changed: ' + name)


def _stage_case(copies, phase, identity):
    original = copies.source / 'runs' / identity
    inputs = copies.source / 'observations' / identity
    staged, observations = m._slots(identity, True)
    execution = json.loads(m._read(original / m.EXECUTION_FILE, copies.source))
    for name in [m.EXECUTION_FILE, 'result.json'] + [e['path'] for e in execution['artifacts']]:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe replay member')
        copies.copy(original / name, staged / name)
    for path in inputs.glob('*.json'):
        copies.copy(path, observations / path.name)
    if '-v2-' not in identity:
        return
    for role in ('before', 'after'):
        for index in range(1, 6):
            name = f'snapshot{index}.stdout'
            copies.copy(copies.snapshots / f'{role}-{phase}' / name,
                        observations / 'telemetry' / f'{role}-{phase}' / name)


def _judge(phase, identity):
    staged, inputs = m._slots(identity, True)
    discovery = m._load(inputs / 'discovery.json', inputs) if phase == 'B' else None
    return m._judgment(phase, identity, staged, inputs, discovery)


def _conclusion(result):
    return {k: result[k] for k in ('verdict', 'reason')}


def _positives(expected):
    results = []
    for phase, identity in CASES:
        try:
            actual = _conclusion(_judge(phase, identity))
        except (ValueError, KeyError, TypeError) as exc:
            results.append({'attempt': identity, 'expected': expected[identity],
                            'reader_error': str(exc), 'matches': False})
            continue
        results.append({'attempt': identity, 'expected': expected[identity],
                        'actual': actual, 'matches': actual == expected[identity]})
    return results


def _negative(label, phase, identity, expected):
    staged, inputs = m._slots(identity, True)
    telemetry = label == 'wrong_telemetry_role'
    path = inputs / 'telemetry/before-B/snapshot1.stdout' if telemetry else staged / 'stdout.bin'
    old = path.read_bytes()
    changed = ((inputs / 'telemetry/after-B/snapshot1.stdout').read_bytes() if telemetry
               else bytes([old[0] ^ 1]) + old[1:])
    required = 'dram_raw_binding' if telemetry else 'capture_artifact'
    path.write_bytes(changed)
    try:
        try:
            reason = _judge(phase, identity)['reason']
        except ValueError as exc:
            reason = str(exc)
        if reason != required:
            raise ValueError('replay negative failed: ' + label + ': ' + reason)
    finally:
        path.write_bytes(old)
    if _conclusion(_judge(phase, identity)) != expected[identity]:
        raise ValueError('restored positive failed: ' + label)
    return {'mutation': label, 'expected_refusal': required, 'actual_refusal': reason}


def replay(source, snapshots, output, baseline):
    output.mkdir(mode=0o700, exist_ok=False)
    copies, original_work = Copies(source, snapshots), m.WORK
    try:
        m.WORK = output / 'bundle'
        for phase, identity in CASES:
            _stage_case(copies, phase, identity)
        expected = {r['attempt']: r['current_evaluate'] for r in baseline['results']}
        results = _positives(expected)
        m.publish(output / 'positive-replay.json', {'results': results})
        if any(not r['matches'] for r in results):
            raise ValueError('baseline replay disagreement; see positive-replay.json')
        negatives = [_negative('stdout_one_byte', *CASES[0], expected),
                     _negative('wrong_telemetry_role', *CASES[3], expected)]
        record = {'results': results, 'negatives': negatives,
                  'historical_difference': 'A v1 originally rejected/watcher; current baseline accepts discovery. '
                  'This is the prospective correction in ADR-0021:57-63,80-84, not a rewritten record '
                  'or a historical-policy branch.', 'recording_process_timezone': os.environ.get('TZ'),
                  'protected_original_sha256': copies.protected}
        m.publish(output / 'replay-result.json', record)
        return record
    finally:
        m.WORK = original_work
        copies.verify_originals()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'snapshots', 'output', 'baseline'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    # Metadata decoder only: these retained snapshots were recorded in JST.
    # This changes this Python process, never host timezone/settings or policy.
    os.environ['TZ'] = 'Asia/Tokyo'
    time.tzset()
    record = replay(args.source.absolute(), args.snapshots.absolute(), args.output.absolute(),
                    json.loads(args.baseline.read_bytes()))
    print(json.dumps({'matches': len(record['results']), 'refused_negatives': len(record['negatives'])}))


if __name__ == '__main__':
    main()
