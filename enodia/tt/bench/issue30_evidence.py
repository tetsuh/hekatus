"""Raw telemetry provenance and conservative native capture bracketing; no IO commands."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime

from defusedxml import ElementTree as ET

from enodia.tt.bench import issue30_admission as admission


def _integer(value):
    if type(value) is int and value >= 0:
        return value
    if type(value) is str:
        value = value.strip(' ')  # tt-smi display padding only; never tabs/newlines.
        if re.fullmatch(r'(?:0x[0-9a-fA-F]+|\d+)', value, re.ASCII):
            return int(value, 16 if value.startswith('0x') else 10)
    raise ValueError('dram_raw_integer')


def _temperature(value):
    # Temperature is a decimal observation, never an integer counter or health threshold.
    if type(value) is not str or not re.fullmatch(r'\d+(?:\.\d+)?', value.strip(' '), re.ASCII):
        raise ValueError('raw_temperature')
    result = float(value.strip(' '))
    if not math.isfinite(result):
        raise ValueError('raw_temperature')
    return result


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('dram_duplicate_key')
        value[key] = item
    return value


def _device(device, identities):
    board, firmware = device['board_info'], device['firmwares']
    telemetry, registers = device['telemetry'], device['smbus_telem']
    identity = identities.get(board['board_id'], {})
    if (board['bus_id'] != identity.get('pci')
            or firmware['fw_bundle_version'] != identity.get('bundle')
            or firmware['cm_fw'] != identity.get('cmfw')):
        raise ValueError('dram_raw_identity')
    ddr = registers['DDR_STATUS']
    card = {'serial': board['board_id'], 'dram_status': board['dram_status'],
            'raw_ddr_status': ddr.strip(' ') if type(ddr) is str else ddr,
            'uncorrectable_errors': _integer(registers['GDDR_UNCORR_ERRS']),
            'arc_heartbeat': _integer(telemetry['heartbeat']),
            'aiclk_mhz': _integer(telemetry['aiclk'])}
    return card, _temperature(telemetry['asic_temperature'])


def _aggregate(env, normalized, temperatures):
    for card in env['cards']:
        if ('temperature_c' in card and (type(card['temperature_c']) not in (int, float)
                or temperatures[card['serial']] != card['temperature_c'])):
            raise ValueError('raw_temperature_summary')
        first = next(c['arc_heartbeat'] for c in normalized[0]['cards'] if c['serial'] == card['serial'])
        last = next(c['arc_heartbeat'] for c in normalized[-1]['cards'] if c['serial'] == card['serial'])
        if card['arc_heartbeat'] != {'first': first, 'second': last}:
            raise ValueError('dram_raw_arc_summary')


def _snapshot_time(value):
    # tt-smi emits host-local naive ISO time. Portable replay restores the
    # recording process timezone; this decoder does not set host configuration.
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}', value, re.ASCII):
        raise ValueError('dram_snapshot_timestamp')
    return datetime.fromisoformat(value).timestamp()


def series(env, phase, role, read):
    """The caller supplies a canonical attempt/role reader, never a hash lookup."""
    if env.get('attempt') != env['owner_unit'][phase] or role not in ('before', 'after'):
        raise ValueError('dram_attempt_role')
    expected = env['dram_admission']['samples']
    if len(expected) != 5:
        raise ValueError('dram_gate_five_required')
    normalized, temperatures = [], {}
    identities = {c['serial']: c for c in env['cards']}
    for index, sample in enumerate(expected, 1):
        raw = read(f'telemetry/{role}-{phase}/snapshot{index}.stdout')
        if hashlib.sha256(raw).hexdigest() != sample['raw_sha256']:
            raise ValueError('dram_raw_binding')
        value = json.loads(raw, object_pairs_hook=_object)
        cards = []
        for device in value['device_info']:
            card, temperature = _device(device, identities)
            cards.append(card)
            temperatures[card['serial']] = temperature
        if sorted(cards, key=lambda c: c['serial']) != sorted(sample['cards'], key=lambda c: c['serial']):
            raise ValueError('dram_raw_values')
        normalized.append(sample | {'cards': cards, 'monotonic_seconds': _snapshot_time(value['time']),
                                    'reported_monotonic_seconds': sample['monotonic_seconds']})
    _aggregate(env, normalized, temperatures)
    # Actual raw dates/values, not summary timestamps, establish spacing/ARC.
    admission.dram_gate(env | {'dram_admission': {'schema': 'issue30-dram-five-v1', 'samples': normalized}})
    if len({s['raw_sha256'] for s in normalized}) != 5:
        raise ValueError('dram_snapshot_reuse')
    return normalized


def _progress(previous, sample):
    old = {c['serial']: c['arc_heartbeat'] for c in previous['cards']}
    new = {c['serial']: c['arc_heartbeat'] for c in sample['cards']}
    if (sample['monotonic_seconds'] <= previous['monotonic_seconds']
            or sample.get('reported_monotonic_seconds', sample['monotonic_seconds'])
            <= previous.get('reported_monotonic_seconds', previous['monotonic_seconds'])
            or old.keys() != new.keys() or any(new[s] <= old[s] for s in old)):
        raise ValueError('dram_series_order')


def ordered(*groups):
    seen, previous = set(), None
    for group in groups:
        for sample in group:
            if sample['raw_sha256'] in seen:
                raise ValueError('dram_snapshot_reuse')
            seen.add(sample['raw_sha256'])
            if previous is not None:
                _progress(previous, sample)
            previous = sample


def _sample_clock(group, boundary, offset, before):
    for sample in group:
        stamp = sample['reported_monotonic_seconds']
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or sample['monotonic_seconds'] > stamp + offset
                or (stamp >= boundary if before else stamp <= boundary)):
            raise ValueError('dram_summary_clock')


def _clock_bounds(clock, execution, native, before, after):
    fields = ('start_wall', 'end_wall', 'start_monotonic', 'end_monotonic')
    if (type(clock) is not dict or clock.get('schema') != 'issue30-capture-clock-v1'
            or clock.get('attempt') != execution.get('attempt')
            or any(type(clock.get(k)) not in (int, float) or not math.isfinite(clock[k]) for k in fields)):
        raise ValueError('dram_capture_clock')
    lower, upper = clock['start_wall'], clock['end_wall']
    start, end = clock['start_monotonic'], clock['end_monotonic']
    if not lower <= native <= upper or end - start != execution['elapsed_seconds']:
        raise ValueError('dram_capture_clock')
    _sample_clock(before, start, lower - start, True)
    _sample_clock(after, end, lower - start, False)
    return lower, upper


def bracket(before, after, xml, execution, clock=None):
    """Native XML time is inside capture; total elapsed conservatively bounds it.

    Exact manifest-bound clock observations narrow that enclosure when present.
    No file mtimes, fabricated historical events or old-policy profiles are used.
    """
    root = ET.fromstring(xml, forbid_dtd=True)
    stamp = root.get('timestamp', '')
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}', stamp):
        raise ValueError('dram_capture_window')
    native = datetime.fromisoformat(stamp).replace(tzinfo=UTC).timestamp()
    elapsed = execution.get('elapsed_seconds')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 60:
        raise ValueError('dram_capture_window')
    lower, upper = native - elapsed, native + elapsed
    if clock is not None:
        lower, upper = _clock_bounds(clock, execution, native, before, after)
    if (before[-1]['monotonic_seconds'] >= lower or after[0]['monotonic_seconds'] <= upper):
        raise ValueError('dram_capture_window')
    ordered(before, after)
