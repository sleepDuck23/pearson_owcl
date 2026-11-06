"""Empirical calibration on observations disjoint from evaluation streams.

These choices fill in values not supplied by the papers; they are not claimed
to recover the authors' original calibration. No evaluation scores enter here.
"""

import copy
import numpy as np

from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD


def make_model(name, n, m, k_conf, detection, novelty, reg_lambda=0.1):
    common = dict(n=n, m=m, k_conf=k_conf, gamma=0.1, factor=1.0,
                  dynamic_gamma=True)
    if name == "MMD":
        return OWR_CPD(eta=detection, nu=novelty, **common)
    if name == "Pearson":
        return PearsonOWRCPD(tau=detection, nu=novelty,
                             reg_lambda=reg_lambda, normalize_distance=False,
                             **common)
    raise ValueError(f"Unknown model: {name}")


def calibrate_thresholds(sample, labels, *, n, m, k_conf, reg_lambda=0.1,
                         seed=10001, repeats=12, names=("MMD", "Pearson")):
    """Calibrate directed comparisons using independent homogeneous batches.

    sample(label, size, rng) returns fresh calibration observations. For each
    reference, R, the matching analysis window and matching recognition batch
    are sliced from one draw, so those observations are disjoint. A reference's
    bandwidth is learned before its embedding/operator and held fixed during
    monitoring. Recognition always uses the stored candidate's bandwidth.

    Detection brackets the 99th percentile of same-law confirmation-run minima
    and the 5th percentile of different-law confirmation-run minima. Recognition
    brackets the corresponding 95% same-law and 5% different-law tails. On
    overlap, keep the same-law boundary and report that no separating interval
    was found. This is finite-sample empirical calibration, not a guarantee.
    """
    labels = tuple(labels)
    if len(labels) < 2 or repeats < 2:
        raise ValueError("Calibration needs at least two regimes and two repeats")
    rng = np.random.RandomState(seed)
    values = {name: {key: [] for key in ("null_detection", "change_detection",
                                        "same_recognition", "other_recognition")}
              for name in names}
    tail_length = m + k_conf - 1

    for _ in range(repeats):
        for label in labels:
            observations = np.asarray(sample(label, 2 * n + tail_length, rng))
            if len(observations) != 2 * n + tail_length or not np.isfinite(observations).all():
                raise ValueError("Calibration sampler must return the requested finite observations")
            X = observations[:n]
            null_tail = observations[n:n + tail_length]
            matching = observations[n + tail_length:]
            alternatives = {other: np.asarray(sample(other, n + tail_length, rng))
                            for other in labels if other != label}

            for name in names:
                # Bounds prevent calibration from triggering alarms or recall.
                bound = 2.0 if name == "MMD" else 1.0 / reg_lambda
                reference = make_model(name, n, m, k_conf, bound, 0.5, reg_lambda)
                for x in X:
                    reference.update(x)
                representative = reference.regimes[0]

                def detection_minimum(tail):
                    monitor = copy.deepcopy(reference)
                    scores = [monitor.update(x) for x in tail]
                    return float(np.min(scores[-k_conf:]))

                def recognition(batch):
                    if name == "MMD":
                        return reference._compute_coherence(batch, representative)
                    return reference._compute_pearson_divergence(batch, representative)

                scores = values[name]
                scores["null_detection"].append(detection_minimum(null_tail))
                scores["same_recognition"].append(recognition(matching))
                for other in alternatives.values():
                    scores["change_detection"].append(detection_minimum(other[n:]))
                    scores["other_recognition"].append(recognition(other[:n]))

    result = {}
    for name, scores in values.items():
        if not all(np.isfinite(observations).all() for observations in scores.values()):
            raise FloatingPointError(f'{name} calibration produced non-finite scores')
        null_high = float(np.quantile(scores["null_detection"], 0.99))
        change_low = float(np.quantile(scores["change_detection"], 0.05))
        detection = (null_high + change_low) / 2 if null_high < change_low else null_high
        if name == "MMD":
            same = float(np.quantile(scores["same_recognition"], 0.05))
            other = float(np.quantile(scores["other_recognition"], 0.95))
            separated = other < same
        else:
            same = float(np.quantile(scores["same_recognition"], 0.95))
            other = float(np.quantile(scores["other_recognition"], 0.05))
            separated = same < other
        novelty = (same + other) / 2 if separated else same
        result[name] = dict(detection=detection, novelty=novelty,
                            detection_separated=null_high < change_low,
                            recognition_separated=separated,
                            null_detection=null_high, change_detection=change_low,
                            same_recognition=same, other_recognition=other,
                            seed=seed, repeats=repeats)
    return result


def print_calibration(calibration, overrides=None):
    overrides = overrides or {}
    for name, values in calibration.items():
        print(f"{name} independent calibration (seed={values['seed']}, repeats={values['repeats']}):")
        for key in ("detection", "novelty"):
            fixed = overrides.get(name, {}).get(key)
            value = values[key] if fixed is None else fixed
            source = "calibrated" if fixed is None else "paper setting"
            print(f"  {key}: {value:.12g} ({source})")
        for key in ("detection", "recognition"):
            if not values[f"{key}_separated"]:
                print(f"  {key}: calibration tails overlap; separation is not assured")
