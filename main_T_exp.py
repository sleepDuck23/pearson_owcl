import numpy as np
from main_mmd_pearson import (
    detection_indices, get_regime_sequence, calculate_delays, print_detection_summary,
)
import matplotlib.pyplot as plt

# Import all four OWR models 
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD       
from model.owcl_p_coh import HybridOWRCPD          
from model.owcl_mah import MahalanobisOWRCPD       



def generate_moment_matched_stream(tier="T1", segment_length=700, seed=42):
    """
    Generates the stream for Experiment 5.2: Robustness under matched moments.
    Four regimes: (1) Gaussian, (2) Laplace, (3) Exponential, (4) Uniform.
    """
    np.random.seed(seed)
    
    # Prescribed sequence of 12 segments
    sequence_ids = [1, 2, 3, 4, 2, 1, 4, 3, 1, 4, 2, 3] 
    
    # Tier parameters
    if tier == "T1": # Distinct means, common variance
        means = {1: 0.0, 2: 3.0, 3: -3.0, 4: 6.0}
        stds = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    elif tier == "T2": # Common mean, distinct variances
        means = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        stds = {1: 1.0, 2: 2.0, 3: 0.5, 4: 3.0}
    elif tier == "T3": # Common mean, common variance
        means = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        stds = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    else:
        raise ValueError("Tier must be 'T1', 'T2', or 'T3'")
        
    stream = []
    actual_cps = []
    current_len = 0
    
    for idx, dist_id in enumerate(sequence_ids):
        target_mean = means[dist_id]
        target_std = stds[dist_id]
        
        if dist_id == 1: # Gaussian
            data = np.random.normal(target_mean, target_std, segment_length)
            
        elif dist_id == 2: # Laplace
            # Laplace variance is 2 * scale^2. So scale = std / sqrt(2)
            scale = target_std / np.sqrt(2)
            data = np.random.laplace(target_mean, scale, segment_length)
            
        elif dist_id == 3: # Exponential
            # Exp variance is scale^2. Mean is scale. 
            data = np.random.exponential(scale=target_std, size=segment_length)
            data = data - target_std + target_mean # Shift to target mean
            
        elif dist_id == 4: # Uniform
            # Uniform variance is (high-low)^2 / 12. Half-width = std * sqrt(3)
            half_width = target_std * np.sqrt(3)
            data = np.random.uniform(target_mean - half_width, target_mean + half_width, segment_length)
            
        stream.extend(data)
        current_len += segment_length
        if idx < len(sequence_ids) - 1:
            actual_cps.append(current_len)
            
    return np.array(stream), actual_cps, sequence_ids

def main():
    # ==========================================
    # EXPERIMENT SETUP (Choose T1, T2, or T3)
    # ==========================================
    TIER = "T1"  
    
    # Experiment parameters defined by the paper Table 1
    configs = {
        "T1": {"n": 150, "m": 40,  "mmd_nu": 0.63, "seg_len": 700},
        "T2": {"n": 300, "m": 100, "mmd_nu": 0.99, "seg_len": 1000},
        "T3": {"n": 600, "m": 200, "mmd_nu": 1.00, "seg_len": 2000},
    }
    
    n = configs[TIER]["n"]
    m = configs[TIER]["m"]
    k_conf = 6 
    
    # We use dynamic gamma (median heuristic) as described in the paper
    gamma = 0.01 
    dynamic_gamma = True
    
    print(f"Generating Open-World stream for Tier {TIER}...")
    data, actual_cps, true_seq = generate_moment_matched_stream(tier=TIER, segment_length=configs[TIER]["seg_len"])
    total_length = len(data)
    
    # Initialize Models
    mmd_model = OWR_CPD(
        n=n, m=m, eta=0.2, nu=configs[TIER]["mmd_nu"], 
        k_conf=k_conf, gamma=gamma,
        dynamic_gamma=dynamic_gamma
    )
    
    pearson_model = PearsonOWRCPD(
        n=n, m=m, tau=0.40, nu=0.48, 
        k_conf=k_conf, reg_lambda=1e-1, gamma=gamma,
        factor=1.0,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )

    hybrid_model = HybridOWRCPD(
        n=n, m=m, tau=0.40, nu=0.90, 
        k_conf=k_conf, reg_lambda=1e-1, gamma=gamma,
        factor=1.0,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )

    mahalanobis_model = MahalanobisOWRCPD(
        n=n, m=m, tau=0.65, nu=0.08, 
        k_conf=k_conf, reg_lambda=1e-1, gamma=gamma,
        factor = 1.0,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )

    # Run Simulation
    print(f"Starting concurrent monitoring (Total steps: {total_length})...")
    mmd_scores, pearson_scores, hybrid_scores, mah_scores = [], [], [], []
    
    for t in range(total_length):
        mmd_scores.append(mmd_model.update(data[t]))
        pearson_scores.append(pearson_model.update(data[t]))
        hybrid_scores.append(hybrid_model.update(data[t]))
        mah_scores.append(mahalanobis_model.update(data[t]))

    # Report
    print("\n" + "="*50)
    print(f"  SIMULATION SUMMARY REPORT: TIER {TIER}")
    print("="*50)
    print(f"\n GROUND TRUTH Sequence: {' -> '.join(map(str, true_seq))}")
    print(f" True Change Points   : {actual_cps}")

    models_summary = [
        ("MMD OWR MODEL", mmd_model),
        ("PEARSON OWR MODEL", pearson_model),
        ("HYBRID OWR MODEL", hybrid_model),
        ("MAHALANOBIS OWR MODEL", mahalanobis_model)
    ]

    for name, model in models_summary:
        print(f"\n {name}")
        print_detection_summary(actual_cps, model)
    print("="*50)

    # Visualization
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    
    axes[0].plot(data, color="black", linewidth=0.5, alpha=0.8, label=f"Tier {TIER} Stream")
    for cp in actual_cps:
        axes[0].axvline(x=cp, color="blue", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Open-World Stream: Robustness under Matched Moments ({TIER})")
    axes[0].legend(loc="upper right")

    model_data = [
        (mmd_scores, mmd_model, "MMD", "teal"),
        (pearson_scores, pearson_model, "Pearson", "purple"),
        (hybrid_scores, hybrid_model, "Hybrid", "crimson"),
        (mah_scores, mahalanobis_model, "Mahalanobis", "darkmagenta")
    ]

    for ax, (scores, model, name, color) in zip(axes[1:], model_data):
        ax.plot(scores, color=color, linewidth=1.2, label=f"{name} Score")
        
        # Plot detection threshold line (eta for MMD, tau for Pearson-based)
        thresh = getattr(model, 'eta', getattr(model, 'tau', 0.1))
        ax.axhline(y=thresh, color="orange", linestyle="--", label="Detection Threshold")
        
        for cp in detection_indices(model):
            ax.axvline(x=cp, color="red", linestyle="-", linewidth=1.5)
        ax.set_title(f"{name} OWR-CPD Model")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
