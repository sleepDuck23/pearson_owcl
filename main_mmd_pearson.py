import numpy as np
import matplotlib.pyplot as plt

# Import the data generator and all four OWR models
from data.generation import generate_regime_stream
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD       
from model.owcl_p_coh import HybridOWRCPD
from model.owcl_mah import MahalanobisOWRCPD       

def get_regime_sequence(regime_log: list) -> str:
    """Preserve assignments after transitions, including recall of the same ID."""
    seq = []
    previous = None
    for regime in regime_log:
        if regime != -1 and regime != previous:
            seq.append(regime)
        previous = regime
    return " -> ".join(map(str, seq)) if seq else "None"


def detection_indices(model):
    """Convert the models' one-based sample counts to zero-based stream indices."""
    return [int(t) - 1 for t in model.global_changepoints]


def evaluate_changepoints(true_cps, detected_cps, max_delay=None):
    """One-to-one causal matching; both inputs use zero-based sample indices.

    Match the first alarm at/after each change and strictly before the next
    change. Additional alarms in that regime are false alarms. max_delay can
    optionally impose a stricter evaluation horizon; None uses the complete
    regime interval, including the remainder of the final regime.
    """
    truth = sorted(int(cp) for cp in true_cps)
    if len(set(truth)) != len(truth) or any(cp < 0 for cp in truth):
        raise ValueError("True changes must be distinct nonnegative indices")
    if max_delay is not None and (not np.isfinite(max_delay) or max_delay < 0):
        raise ValueError("max_delay must be nonnegative and finite")
    matched = set()
    matches, false_alarms = [], []
    for detection in sorted(int(cp) for cp in detected_cps):
        i = int(np.searchsorted(truth, detection, side="right")) - 1
        if (i >= 0 and i not in matched
                and (max_delay is None or detection - truth[i] <= max_delay)):
            matched.add(i)
            matches.append((truth[i], detection))
        else:
            false_alarms.append(detection)
    return {
        "matches": matches,
        "delays": [alarm - change for change, alarm in matches],
        "false_alarms": false_alarms,
        "missed_changes": [cp for i, cp in enumerate(truth) if i not in matched],
    }


def calculate_delays(true_cps: list, detected_cps: list) -> list:
    """Matched delays only; use evaluate_changepoints for false alarms/misses."""
    return evaluate_changepoints(true_cps, detected_cps)["delays"]


def print_detection_summary(true_cps, model):
    cps = detection_indices(model)
    result = evaluate_changepoints(true_cps, cps)
    delays = result["delays"]
    mean_delay = f"{np.mean(delays):.2f} steps" if delays else "N/A (no matched detections)"
    print(f"  Detected Changes : {cps} (zero-based)")
    print(f"  Matched Changes  : {len(result['matches'])} / {len(true_cps)}")
    print(f"  False Alarms     : {result['false_alarms']}")
    print(f"  Missed Changes   : {result['missed_changes']}")
    print(f"  Detection Delays : {delays}")
    print(f"  Mean Delay       : {mean_delay}")
    print(f"  Total Regimes    : {len(model.regimes)}")
    print(f"  Regime Sequence  : {get_regime_sequence(model.regime_log)}")


def main():
    # 1. Generate the stream data (A -> B -> C -> B -> A -> D)
    total_length = 6000
    true_cps = [1000, 2000, 3000, 4000, 5000]  # True change points
    n = 150        # Reference window size
    m = 50         # Test window size
    gamma = 0.6    # RBF kernel bandwidth
    k_conf = 6     # Consecutive windows to confirm change
    dynamic_gamma = True  # Use dynamic gamma adjustment
    
    print("Generating Open-World stream data...\n")
    data, actual_cps = generate_regime_stream(
        length=total_length, 
        changepoints=true_cps, 
        means=[0.0, 5.0, -4.0,  5.0, 0.0, 10.0],  
        stds=[1.0, 1.0, 0.5, 1.0, 1.0, 2.0],
        seed=42
    )

    # 2. Initialize all four models
    mmd_model = OWR_CPD(
        n=n, 
        m=m, 
        eta=0.1, 
        nu=0.90, 
        k_conf=k_conf, 
        gamma=gamma,
        dynamic_gamma=dynamic_gamma
    )
    
    pearson_model = PearsonOWRCPD(
        n=n, 
        m=m, 
        tau=0.60,  
        nu=0.50,  
        k_conf=k_conf, 
        reg_lambda=1e-5, 
        gamma=gamma,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )

    hybrid_model = HybridOWRCPD(
        n=n, 
        m=m, 
        tau=0.60,  
        nu=0.90,  
        k_conf=k_conf, 
        reg_lambda=1e-5, 
        gamma=gamma,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )

    mahalanobis_model = MahalanobisOWRCPD(
        n=n, 
        m=m, 
        tau=0.60,  
        nu=0.08,  
        k_conf=k_conf, 
        reg_lambda=1e-5, 
        gamma=gamma,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )

    # 3. Run the online simulation
    print("Starting concurrent monitoring...")
    mmd_scores = []
    pearson_scores = []
    hybrid_scores = []
    mah_scores = []
    
    for t in range(total_length):
        mmd_scores.append(mmd_model.update(data[t]))
        pearson_scores.append(pearson_model.update(data[t]))
        hybrid_scores.append(hybrid_model.update(data[t]))
        mah_scores.append(mahalanobis_model.update(data[t]))

    # 4. Readable Summary Report
    print("\n" + "="*50)
    print("  SIMULATION SUMMARY REPORT")
    print("="*50)

    print("\n GROUND TRUTH")
    print(f"  Change Points      : {actual_cps}")
    print(f"  Regime Sequence    : A -> B -> C -> B -> A -> D")

    for name, model in (("MMD", mmd_model), ("PEARSON", pearson_model),
                        ("HYBRID", hybrid_model), ("MAHALANOBIS", mahalanobis_model)):
        print(f"\n {name} OWR MODEL")
        print_detection_summary(actual_cps, model)
    print("\n" + "="*50)

    # 5. Visualization and Comparison (5 subplots)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    # --- Panel 1: Input Signal ---
    ax1.plot(data, color="black", linewidth=0.8, alpha=0.7, label="Stream Data")
    for i, cp in enumerate(actual_cps):
        label = "True Change" if i == 0 else ""
        ax1.axvline(x=cp, color="blue", linestyle="--", linewidth=2, label=label)
    
    ax1.set_title("Open-World Stream (Mean & Variance Shifts)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: MMD Model Score ---
    ax2.plot(mmd_scores, color="teal", linewidth=1.5, label="MMD Score")
    ax2.axhline(y=mmd_model.eta, color="orange", linestyle="--", linewidth=2, label=rf"Threshold $\eta$ = {mmd_model.eta}")
    for i, cp in enumerate(detection_indices(mmd_model)):
        label = "MMD Detection" if i == 0 else ""
        ax2.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
    ax2.set_title("OWR-CPD (MMD) Model")
    ax2.set_ylabel("MMD Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Pearson Model Score ---
    ax3.plot(pearson_scores, color="purple", linewidth=1.5, label="Pearson Divergence")
    ax3.axhline(y=pearson_model.tau, color="green", linestyle="--", linewidth=2, label=rf"Threshold $\tau$ = {pearson_model.tau}")
    for i, cp in enumerate(detection_indices(pearson_model)):
        label = "Pearson Detection" if i == 0 else ""
        ax3.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
    ax3.set_title("Pearson OWR-CPD Model")
    ax3.set_ylabel("Pearson Score")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Hybrid Model Score ---
    ax4.plot(hybrid_scores, color="crimson", linewidth=1.5, label="Hybrid Score (Pearson CPD)")
    ax4.axhline(y=hybrid_model.tau, color="darkorange", linestyle="--", linewidth=2, label=rf"Threshold $\tau$ = {hybrid_model.tau}")
    for i, cp in enumerate(detection_indices(hybrid_model)):
        label = "Hybrid Detection" if i == 0 else ""
        ax4.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
    ax4.set_title("Hybrid OWR-CPD Model")
    ax4.set_ylabel("Hybrid Score")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Mahalanobis Model Score ---
    ax5.plot(mah_scores, color="darkmagenta", linewidth=1.5, label="Mahalanobis Score (Pearson CPD)")
    ax5.axhline(y=mahalanobis_model.tau, color="goldenrod", linestyle="--", linewidth=2, label=rf"Threshold $\tau$ = {mahalanobis_model.tau}")
    for i, cp in enumerate(detection_indices(mahalanobis_model)):
        label = "Mahalanobis Detection" if i == 0 else ""
        ax5.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
    ax5.set_title("Mahalanobis OWR-CPD Model")
    ax5.set_xlabel("Time Step (t)")
    ax5.set_ylabel("Detection Score")
    ax5.legend(loc="upper left")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
