"""Causal detection matching and event-based regime recognition metrics."""

import numpy as np


def get_regime_sequence(regime_log):
    seq, previous = [], None
    for regime in regime_log:
        if regime != -1 and regime != previous:
            seq.append(regime)
        previous = regime
    return " -> ".join(map(str, seq)) if seq else "None"


def detection_indices(model):
    """Convert one-based alarm sample counts to zero-based stream indices."""
    return [int(t) - 1 for t in model.global_changepoints]


def evaluate_changepoints(true_cps, detected_cps, max_delay=None):
    """Match at most one alarm per change, before the next change and deadline.

    Inputs index the first changed sample and alarm sample, both zero-based.
    Delays count post-change observations: alarm at the first new sample has
    delay 1, equal to t_alarm - t0 with the papers' last-old-sample definition.
    """
    truth = sorted(int(cp) for cp in true_cps)
    if len(set(truth)) != len(truth) or any(cp < 0 for cp in truth):
        raise ValueError("True changes must be distinct nonnegative indices")
    if max_delay is not None and (not np.isfinite(max_delay) or max_delay < 1):
        raise ValueError("max_delay must be positive and finite")
    matched, matches, false_alarms = set(), [], []
    for detection in sorted(int(cp) for cp in detected_cps):
        i = int(np.searchsorted(truth, detection, side="right")) - 1
        if (i >= 0 and i not in matched
                and (max_delay is None or detection - truth[i] + 1 <= max_delay)):
            matched.add(i)
            matches.append((truth[i], detection))
        else:
            false_alarms.append(detection)
    return dict(matches=matches, delays=[alarm - cp + 1 for cp, alarm in matches],
                false_alarms=false_alarms,
                missed_changes=[cp for i, cp in enumerate(truth) if i not in matched])


def calculate_delays(true_cps, detected_cps):
    return evaluate_changepoints(true_cps, detected_cps)["delays"]


def evaluate_recognition(true_labels, model, detection_result):
    """Count returns, including failed detections and impure/missing batches.

    A return succeeds only when its matched alarm yields a homogeneous batch
    within that segment, recalls an existing ID, and uses the first ID created
    for that true regime. New IDs for previously seen regimes are failures.
    """
    labels = np.asarray(true_labels)
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    matches = dict(detection_result["matches"])
    canonical, event_results = {}, {}
    contaminated = 0
    for event in model.recognition_log:
        batch = labels[event['batch_start']:event['batch_end']]
        pure = len(batch) == model.n and np.all(batch == batch[0])
        if not pure:
            contaminated += 1
            continue
        label = batch[0].item()
        known_id = canonical.get(label)
        correct_return = not event['is_new'] and event['regime'] == known_id
        correct_new = event['is_new'] and known_id is None
        if event['is_new']:
            canonical.setdefault(label, event['regime'])
        event_results[event['alarm_time']] = (label, event['time'], correct_return, correct_new)

    result = dict(recall_correct=0, recall_total=0, novel_correct=0, novel_total=0,
                  contaminated_batches=contaminated)
    seen = set()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        label = labels[start].item()
        returning = label in seen
        prefix = "recall" if returning else "novel"
        result[f"{prefix}_total"] += 1
        alarm = matches.get(int(start))
        event = event_results.get(None if start == 0 else alarm + 1 if alarm is not None else -1)
        if event is not None:
            event_label, time, correct_return, correct_new = event
            if event_label == label and time <= end:
                result[f"{prefix}_correct"] += int(correct_return if returning else correct_new)
        seen.add(label)
    return result


def print_detection_summary(true_cps, model, max_delay=None, true_labels=None):
    cps = detection_indices(model)
    result = evaluate_changepoints(true_cps, cps, max_delay)
    delays = result["delays"]
    print(f"  Detected Changes : {cps} (zero-based)")
    print(f"  Detection Horizon: {max_delay if max_delay is not None else 'until next change'}")
    print(f"  Matched Changes  : {len(result['matches'])} / {len(true_cps)}")
    print(f"  False Alarms     : {result['false_alarms']}")
    print(f"  Missed Changes   : {result['missed_changes']}")
    print(f"  Detection Delays : {delays} (t_alarm - t0)")
    print(f"  Mean Delay       : {np.mean(delays):.2f} steps" if delays else "  Mean Delay       : N/A")
    print(f"  Total Regimes    : {len(model.regimes)}")
    print(f"  Regime Sequence  : {get_regime_sequence(model.regime_log)}")
    if true_labels is not None:
        recognition = evaluate_recognition(true_labels, model, result)
        for label, key in (("Re-identification", "recall"), ("Novel Discovery", "novel")):
            print(f"  {label:17}: {recognition[key + '_correct']} / {recognition[key + '_total']}")
        print(f"  Mixed Rep. Batches: {recognition['contaminated_batches']}")
        result.update(recognition)
    return result
