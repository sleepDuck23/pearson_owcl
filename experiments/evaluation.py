"""Shared evaluator. Truth is never passed to either online detector.

Stream indices are zero-based. Detector logs use one-based observation counts;
convert both alarm and decision times exactly once at the evaluator boundary.
"""

import numpy as np


def get_regime_sequence(regime_log):
    seq, previous = [], None
    for regime in regime_log:
        if regime != -1 and regime != previous:
            seq.append(regime)
        previous = regime
    return " -> ".join(map(str, seq)) if seq else "None"


def detection_indices(model):
    """Confirmed alarm indices, never backdated to threshold crossing."""
    return [int(t) - 1 for t in model.global_changepoints]


def validate_max_delay(max_delay):
    if (not isinstance(max_delay, (int, np.integer))
            or isinstance(max_delay, (bool, np.bool_)) or max_delay < 0):
        raise ValueError("max_delay must be an explicit nonnegative integer in samples")


def episode_boundaries(true_labels):
    labels = np.asarray(true_labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Provide one ground-truth label per sample in a nonempty stream")
    return np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]


def evaluate_changepoints(true_cps, detected_cps, max_delay, *, stream_length=None,
                         warmup_end=0):
    """Earliest causal one-to-one match, inclusive delay, exclusive episode end.

    A change is the first sample in the new episode. Delay is alarm - change,
    so confirmation on that sample has delay zero. Initialization observations
    [0, warmup_end) are excluded; every other unmatched alarm is false.
    """
    validate_max_delay(max_delay)
    truth = sorted(int(cp) for cp in true_cps)
    if len(set(truth)) != len(truth) or any(cp < warmup_end for cp in truth):
        raise ValueError("True changes must be distinct indices after initialization")
    alarms = sorted(int(cp) for cp in detected_cps)
    if any(t < 0 for t in alarms):
        raise ValueError("Alarm indices must be nonnegative")
    if stream_length is not None and (any(cp >= stream_length for cp in truth)
                                      or any(t >= stream_length for t in alarms)):
        raise ValueError("Changes and alarms must be within the stream")
    matched, matches, false_alarms, ignored = set(), [], [], []
    for alarm in alarms:
        if alarm < warmup_end:
            ignored.append(alarm)
            continue
        i = int(np.searchsorted(truth, alarm, side="right")) - 1
        if i >= 0 and i not in matched and alarm - truth[i] <= max_delay:
            matched.add(i)
            matches.append((truth[i], alarm))
        else:
            false_alarms.append(alarm)
    delays = [alarm - cp for cp, alarm in matches]
    return dict(matches=matches, delays=delays, false_alarms=false_alarms,
                ignored_warmup_alarms=ignored,
                missed_changes=[cp for i, cp in enumerate(truth) if i not in matched],
                detected_count=len(matches), change_total=len(truth),
                false_alarm_count=len(false_alarms), delay_sum=sum(delays),
                mean_delay=float(np.mean(delays)) if delays else None,
                max_delay=int(max_delay))


def calculate_delays(true_cps, detected_cps, max_delay):
    return evaluate_changepoints(true_cps, detected_cps, max_delay)["delays"]


def evaluate_recognition(true_labels, model, detection_result):
    """Score only new decisions associated with matched alarms.

    Maintain a fixed injective truth-to-ID mapping in decision-time order.
    Unmatched, missing, or late decisions cannot establish a mapping. All ID
    creations are tracked, so a previously created but unmapped ID is not fresh.
    Batch purity is deliberately not an additional scoring requirement.
    """
    labels = np.asarray(true_labels)
    boundaries = episode_boundaries(labels)
    episodes, seen_truth = [], set()
    result = dict(recall_correct=0, recall_total=0, novel_correct=0, novel_total=0)
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        label = labels[start].item()
        returning = label in seen_truth
        episodes.append(dict(start=int(start), end=int(end), label=label,
                             returning=returning, correct=False,
                             alarm=None, assignment_time=None, model_id=None))
        if i:
            result['recall_total' if returning else 'novel_total'] += 1
        seen_truth.add(label)

    by_start = {ep['start']: ep for ep in episodes[1:]}
    by_alarm = {}
    for change, alarm in detection_result['matches']:
        ep = by_start[change]
        ep['alarm'] = alarm
        by_alarm[alarm] = ep

    canonical, owner, created_ids, consumed_alarms = {}, {}, set(), set()
    for event in sorted(model.recognition_log, key=lambda e: e['time']):
        time, model_id = int(event['time']) - 1, int(event['regime'])
        fresh = bool(event['is_new']) and model_id not in created_ids
        created_ids.add(model_id)
        alarm = event['alarm_time']
        if alarm is None:
            if not canonical and 0 <= time < episodes[0]['end']:
                canonical[episodes[0]['label']] = model_id
                owner[model_id] = episodes[0]['label']
            continue
        alarm = int(alarm) - 1
        if alarm in consumed_alarms:
            continue
        consumed_alarms.add(alarm)
        ep = by_alarm.get(alarm)
        if ep is None or not (alarm <= time < ep['end']):
            continue
        ep['assignment_time'], ep['model_id'] = time, model_id
        label = ep['label']
        if label in canonical:
            correct = not event['is_new'] and model_id == canonical[label]
        else:
            correct = fresh and model_id not in owner
            if correct:
                canonical[label], owner[model_id] = model_id, label
        ep['correct'] = bool(correct)
        result['recall_correct' if ep['returning'] else 'novel_correct'] += int(correct)
    result.update(mapping=canonical, episodes=episodes)
    return result


def evaluate_model(true_cps, true_labels, model, max_delay):
    """The sole scoring entry point for every model and benchmark."""
    boundaries = episode_boundaries(true_labels)
    if list(true_cps) != boundaries[1:-1].tolist():
        raise ValueError("True changes must equal the episode-label boundaries")
    result = evaluate_changepoints(
        true_cps, detection_indices(model), max_delay,
        stream_length=len(true_labels), warmup_end=model.n)
    result.update(evaluate_recognition(true_labels, model, result))
    return result


def aggregate_metrics(metrics):
    """Pool counts and matched delays, never average per-run rates or means."""
    if not metrics:
        raise ValueError("Provide at least one run")
    tolerances = {m['max_delay'] for m in metrics}
    if len(tolerances) != 1:
        raise ValueError("Cannot pool runs with different alarm tolerances")
    keys = ('detected_count', 'change_total', 'false_alarm_count', 'delay_sum',
            'novel_correct', 'novel_total', 'recall_correct', 'recall_total')
    result = {key: sum(m[key] for m in metrics) for key in keys}
    result['mean_delay'] = (result['delay_sum'] / result['detected_count']
                            if result['detected_count'] else None)
    result['max_delay'] = tolerances.pop()
    return result


def format_rate(correct, total):
    return f'{correct}/{total} ({100 * correct / total:.1f}%)' if total else f'{correct}/{total} (N/A)'


def comparison_table(metrics):
    """Markdown table suitable for the console and saved benchmark reports."""
    rows = [('Detected changes', lambda m: format_rate(m['detected_count'], m['change_total'])),
            ('False alarms', lambda m: str(m['false_alarm_count'])),
            ('Mean detection delay (samples)', lambda m: 'N/A' if m['mean_delay'] is None
             else f"{m['mean_delay']:.2f}"),
            ('First-encounter discovery', lambda m: format_rate(m['novel_correct'], m['novel_total'])),
            ('Return-episode assignment', lambda m: format_rate(m['recall_correct'], m['recall_total']))]
    lines = ['| Metric | MMD | Pearson |', '|---|---:|---:|']
    for label, value in rows:
        lines.append(f"| {label} | {value(metrics['MMD'])} | {value(metrics['Pearson'])} |")
    return '\n'.join(lines)


def print_detection_summary(true_cps, model, max_delay, true_labels=None):
    """Compatibility summary; benchmark runners use the side-by-side table."""
    result = (evaluate_model(true_cps, true_labels, model, max_delay) if true_labels is not None
              else evaluate_changepoints(true_cps, detection_indices(model), max_delay,
                                         stream_length=model.t, warmup_end=model.n))
    print(f"  Alarm tolerance: {max_delay} samples (inclusive)")
    print(f"  Detected changes: {result['detected_count']}/{result['change_total']}")
    print(f"  False alarms: {result['false_alarm_count']}")
    print(f"  Mean delay: {result['mean_delay'] if result['mean_delay'] is not None else 'N/A'}")
    return result
