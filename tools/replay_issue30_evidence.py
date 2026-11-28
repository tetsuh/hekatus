"""Replay retained Issue30 bytes in private copies; never issue board authority."""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
import time
from pathlib import Path

from enodia.tt.bench import issue30 as m
from enodia.tt.bench import issue30_formats as formats

UNAVAILABLE_DRAM = ('Original twenty telemetry snapshots were lost with private temporary storage '
                    'on reboot after the November 27 archive omitted that storage. '
                    'Raw binding, raw value/clock ordering and raw launch freshness cannot be replayed; '
                    'public derivatives are not substituted.')

CASES = (("A", "issue30-a-discovery-watcher-v1-789de6a8faf8"),
         ("B", "issue30-b-fabric1d-smoke-watcher-v1-82c9b462d5ba"),
         ("A", "issue30-a-discovery-watcher-v2-c1f49b70"),
         ("B", "issue30-b-fabric1d-smoke-watcher-v2-c1f49b70"))
DISCOVERY_FILE = 'discovery.json'
NATIVE_XML_FILE = 'native/gtest.xml'
CONCLUSIONS = ({'verdict': 'accepted_discovery', 'reason': 'accepted'},
               {'verdict': 'rejected', 'reason': 'execution'},
               {'verdict': 'accepted_discovery', 'reason': 'accepted'},
               {'verdict': 'accepted_transfer', 'reason': 'accepted'})


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
    execution = formats.load(m._read(original / m.EXECUTION_FILE, copies.source))
    formats.manifest(execution['artifacts'])
    for name in [m.EXECUTION_FILE, 'result.json'] + [e['path'] for e in execution['artifacts']]:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe replay member')
        copies.copy(original / name, staged / name)
    for path in inputs.glob('*.json'):
        copies.copy(path, observations / path.name)
    if '-v2-' not in identity or copies.snapshots is None:
        return
    for role in ('before', 'after'):
        for index in range(1, 6):
            name = f'snapshot{index}.stdout'
            copies.copy(copies.snapshots / f'{role}-{phase}' / name,
                        observations / 'telemetry' / f'{role}-{phase}' / name)


def _audit_capture(phase, identity):
    """Incomplete read-only audit, deliberately not a live judgment option."""
    output, inputs = m._slots(identity, True)
    execution, review = m._review_capture(phase, identity, output, inputs)
    before, after = m._observations(inputs, review)
    discovery = m._load(inputs / DISCOVERY_FILE, inputs) if phase == 'B' else None
    if phase == 'B' and 'owner_unit' in before:
        aid = before['owner_unit']['A']
        aout, _ = m._slots(aid, True)
        original = m._load(aout / 'result.json', aout)
        m._fresh_result(before, discovery, original, _audit_capture('A', aid))
    result = m._capture_result(phase, identity, before, after, m._native(output, execution),
                               execution, discovery)
    return result | {'xml_present': True}


def _judge(phase, identity):
    staged, inputs = m._slots(identity, True)
    before = m._load(inputs / 'before.json', inputs)
    if 'owner_unit' in before and not (inputs / 'telemetry').exists():
        return _audit_capture(phase, identity) | {'eligible_for_admission': False}
    discovery = m._load(inputs / DISCOVERY_FILE, inputs) if phase == 'B' else None
    if phase == 'B' and 'owner_unit' not in before:
        expected = m._load(m.ROOT / m.OWNER_ACCEPTANCE, m.ROOT)
        if not formats.equal(discovery, expected):
            raise ValueError('owner_acceptance_binding')
    if 'owner_unit' not in before:
        return _audit_capture(phase, identity) | {'eligible_for_admission': False}
    return m._judgment(phase, identity, staged, inputs, discovery) | {'eligible_for_admission': False}


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
        _, inputs = m._slots(identity, True)
        incomplete = '-v2-' in identity and not (inputs / 'telemetry').exists()
        results.append({'attempt': identity, 'expected': expected[identity],
                        'actual': actual, 'matches': formats.equal(actual, expected[identity]),
                        'all_checks_passed': actual['reason'] == 'accepted' and not incomplete,
                        'validation_complete': not incomplete, 'eligible_for_admission': False,
                        'unavailable_checks': [UNAVAILABLE_DRAM] if incomplete else []})
    return results


def _negative(label, phase, identity, expected):
    staged, inputs = m._slots(identity, True)
    telemetry = label == 'wrong_telemetry_role'
    path = inputs / 'telemetry/before-B/snapshot1.stdout' if telemetry else staged / m.STDOUT_FILE
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
    if not formats.equal(_conclusion(_judge(phase, identity)), expected[identity]):
        raise ValueError('restored positive failed: ' + label)
    return {'mutation': label, 'expected_refusal': required, 'actual_refusal': reason}


def _coherent_capture(path, output, inputs):
    execution_path, review_path = output / 'execution.json', inputs / 'review.json'
    saved = {p: p.read_bytes() for p in (execution_path, review_path)}
    execution, review = formats.load(saved[execution_path]), formats.load(saved[review_path])
    name = path.relative_to(output).as_posix()
    entry = next(e for e in execution['artifacts'] if e['path'] == name)
    entry.update({'bytes': path.stat().st_size, 'sha256': m.digest(path.read_bytes())})
    execution_path.write_text(json.dumps(execution))
    review['execution_sha256'] = m.digest(execution_path.read_bytes())
    review_path.write_text(json.dumps(review))
    return saved


def _identity_negative(label, expected):
    phase, identity = CASES[3]
    output, inputs = m._slots(identity, True)
    path = output / m.STDOUT_FILE if label == 'warning_bus' else inputs / DISCOVERY_FILE
    saved = {path: path.read_bytes()}
    if label == 'warning_bus':
        changed, count = re.subn(rb'bus_id=0x[0-9a-fA-F]+', b'bus_id=0xff', saved[path], count=1)
        if count != 1:
            raise ValueError('missing positive bus evidence')
        path.write_bytes(changed)
        saved.update(_coherent_capture(path, output, inputs))
        required = 'diagnostic'
    else:
        discovery = formats.load(saved[path])
        if label == 'discovery_attempt':
            discovery['attempt'] = 'other-attempt'
        else:
            discovery['evidence_sha256']['stdout'] = '0' * 64
        path.write_text(json.dumps(discovery))
        required = 'fresh_A_canonical_result'
    try:
        try:
            reason = _judge(phase, identity)['reason']
        except ValueError as exc:
            reason = str(exc)
        if reason != required:
            raise ValueError('identity negative failed: ' + label + ': ' + reason)
    finally:
        for item, raw in saved.items():
            item.write_bytes(raw)
    if not formats.equal(_conclusion(_judge(phase, identity)), expected[identity]):
        raise ValueError('restored identity positive failed')
    return {'mutation': label, 'actual_refusal': reason, 'expected_refusal': required,
            'derived_fingerprints_updated': label == 'warning_bus'}


def _finding_negative(label, expected):
    phase, identity = CASES[2] if label in ('down_diagnostic', 'native_xml_cdata_A') else CASES[3]
    output, inputs = m._slots(identity, True)
    names = {'watcher_completion': 'logs/generated/watcher/watcher.log',
             'native_xml_exactness': NATIVE_XML_FILE,
             'native_xml_cdata_A': NATIVE_XML_FILE, 'native_xml_cdata_B': NATIVE_XML_FILE,
             'down_diagnostic': m.STDOUT_FILE,
             'canonical_json_types': 'observations-before.json'}
    path = output / names[label]
    saved = {path: path.read_bytes()}
    raw = saved[path]
    required = {'watcher_completion': 'watcher', 'native_xml_exactness': 'test_result',
                'native_xml_cdata_A': 'test_result', 'native_xml_cdata_B': 'test_result',
                'down_diagnostic': 'diagnostic',
                'canonical_json_types': 'captured before observation mismatch'}[label]
    if label == 'watcher_completion':
        changed = b'Dump #1 at 0.919s\n'
    elif label in ('native_xml_cdata_A', 'native_xml_cdata_B'):
        changed = raw.replace(b'</testsuites>', b'<![CDATA[ \t\n]]></testsuites>', 1)
    elif label == 'native_xml_exactness':
        changed = raw.replace(b' />', b'><x:failure xmlns:x="urn:counterexample"/></testcase>', 1)
    elif label == 'down_diagnostic':
        changed, count = re.subn(rb'(\| warning {2}\| +)Test( +\| Chip:)', rb'\g<1>UMD\2', raw, count=1)
        if count != 1:
            raise ValueError('missing positive DOWN evidence')
    else:
        value = formats.load(raw)
        value['cards'][0]['aiclk_mhz'] = float(value['cards'][0]['aiclk_mhz'])
        changed = json.dumps(value).encode()
    if changed == raw:
        raise ValueError('missing positive mutation target')
    path.write_bytes(changed)
    saved.update(_coherent_capture(path, output, inputs))
    try:
        try:
            reason = _judge(phase, identity)['reason']
        except ValueError as exc:
            reason = str(exc)
        if reason != required:
            raise ValueError('finding negative failed: ' + label + ': ' + reason)
    finally:
        for item, original in saved.items():
            item.write_bytes(original)
    if not formats.equal(_conclusion(_judge(phase, identity)), expected[identity]):
        raise ValueError('restored finding positive failed: ' + label)
    return {'mutation': label, 'expected_refusal': required, 'actual_refusal': reason,
            'derived_fingerprints_updated': True, 'original_data_copy': True}


def _legacy_live_negative(phase, identity, expected):
    output, inputs = m._slots(identity, True)
    if not formats.equal(_conclusion(_judge(phase, identity)), expected[identity]):
        raise ValueError('legacy replay control failed')
    discovery = m._load(inputs / DISCOVERY_FILE, inputs) if phase == 'B' else None
    try:
        m._judgment(phase, identity, output, inputs, discovery)
    except ValueError as exc:
        reason = str(exc)
    else:
        raise ValueError('legacy live judgment was admitted')
    if reason != 'dram_live_series_required':
        raise ValueError('legacy live refusal reason: ' + reason)
    if not formats.equal(_conclusion(_judge(phase, identity)), expected[identity]):
        raise ValueError('legacy replay restoration failed')
    return {'mutation': 'legacy_live_' + phase, 'expected_refusal': reason,
            'actual_refusal': reason, 'control': 'historical replay only',
            'eligible_for_admission': False, 'original_data_copy': True}


def _private_root(value):
    candidate = value.absolute()
    temporary = Path(tempfile.gettempdir()).resolve()
    if (candidate.is_relative_to(m.ROOT) or temporary.is_relative_to(m.ROOT)
            or candidate.parent != temporary
            or not re.fullmatch(r'(?:tidd-195-|pi30-)[A-Za-z0-9_-]+', candidate.name, re.ASCII)):
        raise ValueError('replay_private_root')
    root = temporary / candidate.name
    m._owned(root, root)
    mode = root.lstat()
    if (not stat.S_ISDIR(mode.st_mode) or mode.st_uid != os.getuid()
            or stat.S_IMODE(mode.st_mode) != 0o700):
        raise ValueError('replay_private_root')
    return root


def _source_root(value):
    root = m.ROOT / 'out/issue30'
    if value.absolute() != root:
        raise ValueError('replay_source_scope')
    return m._owned(root, root)


def _output_path(value):
    root = _private_root(value.parent)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value.name, re.ASCII):
        raise ValueError('replay_output_scope')
    return m._owned(root / value.name, root)


def _baseline(value):
    if value.name != 'replay-baseline.json':
        raise ValueError('replay_baseline_scope')
    root = _private_root(value.parent)
    value = formats.load(m._read(root / 'replay-baseline.json', root))
    _validate_baseline(value)
    return value


def _validate_baseline(value):
    formats.fields(value, ('results',), ('compared_revision', 'scope', 'original_files_verified_unchanged'),
                   'replay_baseline_schema')
    expected = {identity: conclusion for (_, identity), conclusion in zip(CASES, CONCLUSIONS, strict=True)}
    rows = value['results']
    if type(rows) is not list or len(rows) != 4:
        raise ValueError('replay_baseline_schema')
    for row in rows:
        formats.fields(row, ('attempt', 'current_evaluate'), reason='replay_baseline_schema')
        if type(row['attempt']) is not str:
            raise ValueError('replay_baseline_schema')
    if {row['attempt'] for row in rows} != expected.keys():
        raise ValueError('replay_baseline_schema')
    for row in rows:
        if not formats.equal(row['current_evaluate'], expected[row['attempt']]):
            raise ValueError('replay_baseline_schema')


def replay(source, snapshots, output, baseline):
    _validate_baseline(baseline)
    source = _source_root(source)
    snapshots = _private_root(snapshots) if snapshots is not None else None
    output = _output_path(output)
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
        negatives = [_negative('stdout_one_byte', *CASES[0], expected)]
        if snapshots is not None:
            negatives.append(_negative('wrong_telemetry_role', *CASES[3], expected))
        negatives += [_finding_negative(label, expected) for label in
                      ('watcher_completion', 'native_xml_exactness', 'down_diagnostic', 'canonical_json_types')]
        negatives += [_identity_negative(label, expected) for label in
                      ('warning_bus', 'discovery_attempt', 'discovery_capture_fingerprint')]
        final_fix_negatives = [_legacy_live_negative(phase, identity, expected)
                               for phase, identity in CASES[:2]]
        final_fix_negatives += [_finding_negative(label, expected)
                               for label in ('native_xml_cdata_A', 'native_xml_cdata_B')]
        record = {'schema': 'issue30-offline-audit-v2', 'results': results, 'negatives': negatives,
                  'final_fix_negatives': final_fix_negatives,
                  'validation_complete': snapshots is not None, 'eligible_for_admission': False,
                  'unavailable_dram': UNAVAILABLE_DRAM if snapshots is None else None,
                  'unavailable_negatives': ['Raw role-swap unavailable; launch freshness uses synthetic fixtures']
                  if snapshots is None else [],
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
    for name in ('source', 'output', 'baseline'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--snapshots', type=Path,
                        help='Original raw snapshots only; absent means explicitly incomplete offline audit')
    args = parser.parse_args()
    # Metadata decoder only: these retained snapshots were recorded in JST.
    # This changes this Python process, never host timezone/settings or policy.
    os.environ['TZ'] = 'Asia/Tokyo'
    time.tzset()
    record = replay(args.source.absolute(), args.snapshots.absolute() if args.snapshots else None, args.output.absolute(),
                    _baseline(args.baseline))
    print(json.dumps({'matches': len(record['results']), 'refused_negatives': len(record['negatives']),
                      'validation_complete': record['validation_complete'], 'eligible_for_admission': False}))
    return 0 if record['validation_complete'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
