"""Closed host evidence formats; no device, process or authority operations."""
from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

SHA = r'[a-f0-9]{64}'
STAMP = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}'
LOGGER = re.compile(r'(?P<date>[0-9-]{10} [0-9:.]{12}) \| '
                    r'(?P<level>info     |warning  |debug    |trace    |error    |critical )'
                    r'\| +(?P<component>[A-Za-z0-9_:]+) +\| (?P<body>[^\r\n]+)')


def fields(value, required, optional=(), reason='evidence_schema'):
    if type(value) is not dict or not set(required) <= value.keys() <= set(required) | set(optional):
        raise ValueError(reason)


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def equal(left, right):
    """JSON structural identity, including each scalar's concrete type."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
    if type(left) is list:
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    if type(left) is float and not math.isfinite(left):
        return False
    return type(left) in (str, int, float, bool, type(None)) and left == right


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('evidence_json')
        result[key] = value
    return result


def _constant(_value):
    raise ValueError('evidence_json')


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('evidence_json')
    return result


def load(raw):
    """Only UTF-8 JSON, no BOM, duplicate keys or nonfinite extensions."""
    try:
        text = raw.decode('utf-8') if type(raw) is bytes else raw
        return json.loads(text, object_pairs_hook=unique, parse_constant=_constant, parse_float=_float)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('evidence_json') from exc


def stamp(value, *, microseconds=False, local=False, reason='evidence_timestamp'):
    grammar = STAMP.replace('{3}', '{6}') if microseconds else STAMP
    if type(value) is not str or not re.fullmatch(grammar, value, re.ASCII):
        raise ValueError(reason)
    try:
        date = datetime.fromisoformat(value)
        if date.isoformat(timespec='microseconds' if microseconds else 'milliseconds') != value:
            raise ValueError(reason)
        return date.timestamp() if local else date.replace(tzinfo=UTC).timestamp()
    except (ValueError, OverflowError, OSError) as exc:
        raise ValueError(reason) from exc


def logger(line):
    match = LOGGER.fullmatch(line)
    if match is None:
        return None
    stamp(match['date'].replace(' ', 'T'), reason='diagnostic')
    return match.groupdict()


def environment(env):
    required = ('schema', 'attempt', 'boot', 'image', 'kmd', 'watcher', 'cards', 'links', 'uncabled')
    fields(env, required, ('topology', 'owner_unit', 'dram_admission'), 'environment')
    if type(env['cards']) is not list:
        raise ValueError('environment')
    if 'owner_unit' in env:
        fields(env['owner_unit'], ('date', 'A', 'B'), reason='owner_unit')
    if 'topology' not in env:
        legacy_links(env)
    for card in env['cards']:
        _card_fields(card)


def _card_fields(card):
    required = ('chip', 'pcie_index', 'serial', 'pci', 'bundle', 'cmfw', 'erisc',
                'aiclk_mhz', 'dram_healthy', 'eth_heartbeat_healthy')
    fields(card, required, ('node', 'arc_heartbeat', 'temperature_c', 'unobservable'), 'environment')
    if type(card['chip']) is not int or type(card['pcie_index']) is not int or card['pcie_index'] < 0:
        raise ValueError('environment')
    if 'arc_heartbeat' in card:
        fields(card['arc_heartbeat'], ('first', 'second'), reason='environment')
    if 'temperature_c' in card and not finite(card['temperature_c']):
        raise ValueError('environment')
    if 'unobservable' in card:
        fields(card['unobservable'], (), ('erisc', 'eth_heartbeat_healthy'), 'unobservable_reason')
    if 'node' in card and (type(card['node']) is not str
            or not re.fullmatch(r'/dev/tenstorrent/\d{1,6}', card['node'], re.ASCII)):
        raise ValueError('environment')


def legacy_links(env):
    if type(env['links']) is not list or type(env['uncabled']) is not list:
        raise ValueError('environment')
    for link in env['links']:
        fields(link, ('chip', 'channel', 'core', 'peer', 'peer_core'), reason='environment')
        if (any(type(link[k]) is not int or link[k] < 0 for k in ('chip', 'channel', 'peer'))
                or any(type(link[k]) is not str or not re.fullmatch(r'\d{1,3}-\d{1,3}', link[k], re.ASCII)
                       for k in ('core', 'peer_core'))):
            raise ValueError('environment')
    for pair in env['uncabled']:
        if type(pair) is not list or len(pair) != 2 or any(type(x) is not int or x < 0 for x in pair):
            raise ValueError('environment')


def review(value):
    fields(value, ('schema', 'attempt', 'execution_sha256', 'before_sha256', 'after_sha256', 'watcher_clean'),
           ('watcher_sha256', 'watcher_observation', 'new_diagnostics', 'decision', 'manual_scope', 'stop'),
           'review_schema')
    for name in ('execution_sha256', 'before_sha256', 'after_sha256', 'watcher_sha256'):
        if name in value and (type(value[name]) is not str or not re.fullmatch(SHA, value[name])):
            raise ValueError('review_schema')
    if value['watcher_clean'] is not None and type(value['watcher_clean']) is not bool:
        raise ValueError('review_schema')
    for name in ('decision', 'manual_scope', 'stop'):
        if name in value and type(value[name]) is not str:
            raise ValueError('review_schema')
    if 'new_diagnostics' in value and (type(value['new_diagnostics']) is not list
            or any(type(line) is not str for line in value['new_diagnostics'])):
        raise ValueError('review_schema')
    _watcher_annotation(value.get('watcher_observation'))


def _watcher_annotation(observation):
    if observation is not None and type(observation) is not str:
        fields(observation, ('fresh', 'scan_passed', 'dump_count', 'scope'), reason='review_schema')
        if (type(observation['fresh']) is not bool or type(observation['scan_passed']) is not bool
                or type(observation['dump_count']) is not int or observation['dump_count'] < 0
                or type(observation['scope']) is not str):
            raise ValueError('review_schema')


def manifest(entries):
    if type(entries) is not list:
        raise ValueError('capture_artifact')
    seen = set()
    for entry in entries:
        fields(entry, ('path', 'bytes', 'sha256'), reason='capture_artifact')
        path = entry['path']
        if (type(path) is not str or not re.fullmatch(r'[A-Za-z0-9_./-]+', path, re.ASCII)
                or path.startswith('/') or any(p in ('', '.', '..') for p in path.split('/'))
                or path in seen or type(entry['bytes']) is not int or entry['bytes'] < 0
                or type(entry['sha256']) is not str or not re.fullmatch(SHA, entry['sha256'])):
            raise ValueError('capture_artifact')
        seen.add(path)


def execution(value):
    fields(value, ('schema', 'attempt', 'exit', 'client_exit', 'started', 'elapsed_seconds',
                  'timed_out', 'terminated', 'device_free', 'watcher_clean', 'containers_clean',
                  'image', 'binary_sha256', 'argv', 'environment', 'artifacts'),
           ('inner_deadline_status', 'error_type', 'create_status'), 'execution_schema')
    manifest(value['artifacts'])
    execution_scalars(value)
    if type(value['argv']) is not list or any(type(x) is not str for x in value['argv']):
        raise ValueError('execution_provenance')
    items = value['environment']
    if type(items) is not list or any(type(x) is not str or '=' not in x for x in items):
        raise ValueError('execution_provenance')
    pairs = [x.split('=', 1) for x in items]
    allowed = {'TT_METAL_WATCHER', 'HOME', 'LD_BIND_NOW', 'TMPDIR', 'TT_METAL_RUNTIME_ROOT',
               'TT_METAL_KERNEL_PATH', 'TT_METAL_CACHE', 'TT_METAL_LOGS_PATH', 'TT_VISIBLE_DEVICES'}
    if len({k for k, _ in pairs}) != len(pairs) or {k for k, _ in pairs} != allowed:
        raise ValueError('execution_provenance')
    expected = {'TT_METAL_WATCHER': '1', 'HOME': '/scratch', 'LD_BIND_NOW': '1', 'TMPDIR': '/scratch',
                'TT_METAL_KERNEL_PATH': '/source', 'TT_METAL_CACHE': '/capture/cache',
                'TT_METAL_LOGS_PATH': '/capture/logs', 'TT_VISIBLE_DEVICES': '0000:06:00.0,0000:09:00.0'}
    observed = dict(pairs)
    if (any(observed[k] != v for k, v in expected.items()) or observed['TT_METAL_RUNTIME_ROOT']
            not in ('/capture/runtime', '/opt/venv/lib/python3.12/site-packages/ttnn')):
        raise ValueError('execution_provenance')


def execution_scalars(value):
    for key in ('started', 'timed_out', 'terminated', 'device_free', 'containers_clean'):
        if type(value[key]) is not bool:
            raise ValueError('execution_schema')
    for key in ('exit', 'client_exit'):
        if value[key] is not None and type(value[key]) is not int:
            raise ValueError('execution_schema')
    if (value['watcher_clean'] is not None and type(value['watcher_clean']) is not bool
            or not finite(value['elapsed_seconds']) or value['elapsed_seconds'] < 0):
        raise ValueError('execution_schema')
    _execution_status(value)


def _execution_status(value):
    if 'create_status' in value:
        if value['create_status'] not in ('not_requested', 'pending', 'identified', 'unknown'):
            raise ValueError('execution_schema')
        if value['started'] and value['create_status'] != 'identified':
            raise ValueError('execution')
    if 'error_type' in value:
        raise ValueError('execution')
    statuses = {124: 'timeout_exit', 137: 'sigkill_or_deadline_unidentified'}
    if value.get('inner_deadline_status', statuses.get(value['exit'], 'not_indicated')) != statuses.get(value['exit'], 'not_indicated'):
        raise ValueError('execution_schema')


def verdict(value):
    fields(value, ('schema', 'attempt', 'phase', 'verdict', 'reason', 'scope', 'baseline_sha256',
                   'evidence_sha256', 'topology', 'observed_qsfp', 'firmware_observation'),
           ('packet_observation', 'xml_present'), 'discovery_binding')
    fields(value['evidence_sha256'], ('xml', 'stdout', 'stderr'), reason='discovery_binding')
    hashes = [value['baseline_sha256'], *value['evidence_sha256'].values()]
    if any(type(v) is not str or not re.fullmatch(SHA, v) for v in hashes):
        raise ValueError('discovery_binding')
    if 'xml_present' in value and type(value['xml_present']) is not bool:
        raise ValueError('discovery_binding')
    if value.get('packet_observation') is not None:
        raise ValueError('discovery_binding')  # This validator admits A evidence only.
    fields(value['topology'], ('outcome', 'reason', 'hypothesis'), reason='discovery_binding')
    observation = value['firmware_observation']
    fields(observation, ('reported_versions', 'per_core_versions', 'erisc_health',
                         'eth_heartbeat_health', 'diagnostic_lines', 'truncated'), reason='discovery_binding')
    if (type(observation['truncated']) is not bool
            or not equal(observation['per_core_versions'], [])
            or type(observation['diagnostic_lines']) is not list
            or any(type(line) is not str for line in observation['diagnostic_lines'])):
        raise ValueError('discovery_binding')
    if type(observation['reported_versions']) is not list:
        raise ValueError('discovery_binding')
    for item in observation['reported_versions']:
        fields(item, ('scope', 'version', 'line'), reason='discovery_binding')
        if item['scope'] != 'first_core_only' or any(type(v) is not str for v in item.values()):
            raise ValueError('discovery_binding')


def _xml_values(node):
    if node.text and node.text.strip() or node.tail and node.tail.strip():
        raise ValueError('test_result')
    for key in ('failures', 'errors', 'disabled', 'skipped'):
        if key in node.attrib and node.get(key) != '0':
            raise ValueError('test_result')
    if 'time' in node.attrib:
        value = node.get('time')
        if not re.fullmatch(r'\d+(?:\.\d+)?', value, re.ASCII) or not 0 <= float(value) <= 40:
            raise ValueError('test_result')
    if 'timestamp' in node.attrib:
        stamp(node.get('timestamp'), reason='test_result')


def _xml_text(raw):
    text = raw.decode('utf-8')
    # Extend the prior lexical refusal with XML whitespace, rather than narrowing
    # its scope to the leading declaration and admitting previously refused PIs.
    encoding = re.search(r'<\?xml[^>]*encoding[ \t\r\n]*=[ \t\r\n]*[\'"](?!UTF-8[\'"])[^\'"]+',
                         text, re.IGNORECASE)
    if encoding is not None:
        raise ValueError('test_result')
    return text


def xml(raw, selector):
    if len(raw) > 1024 * 1024:
        raise ValueError('test_result')
    try:
        root = ET.fromstring(_xml_text(raw), forbid_dtd=True)
    except (UnicodeError, ParseError, DefusedXmlException) as exc:
        raise ValueError('test_result') from exc
    suite, name = selector.split('.')
    if (root.tag != 'testsuites' or len(root) != 1 or root[0].tag != 'testsuite'
            or len(root[0]) != 1 or root[0][0].tag != 'testcase' or len(root[0][0]) != 0):
        raise ValueError('test_result')
    common = ('failures', 'errors', 'disabled', 'skipped', 'time', 'timestamp')
    for node in (root, root[0]):
        fields(node.attrib, ('tests',), (*common, 'name'), 'test_result')
        if node.get('tests') != '1':
            raise ValueError('test_result')
    case = root[0][0]
    fields(case.attrib, ('name', 'classname', 'status', 'result'),
           ('file', 'line', 'time', 'timestamp'), 'test_result')
    if (root.get('name', 'AllTests') != 'AllTests' or root[0].get('name') != suite
            or case.get('name') != name or case.get('classname') != suite
            or case.get('status') != 'run' or case.get('result') != 'completed'
            or 'line' in case.attrib and not re.fullmatch(r'\d+', case.get('line'), re.ASCII)):
        raise ValueError('test_result')
    for node in root.iter():
        _xml_values(node)
