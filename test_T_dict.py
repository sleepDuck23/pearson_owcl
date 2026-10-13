import numpy as np
import matplotlib.pyplot as plt

# Import all OWR models 
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD       
from model.owcl_p_coh import HybridOWRCPD          
from model.owcl_mah import MahalanobisOWRCPD       
from model.owcl_dict import DictionaryOWRCPD 

def get_regime_sequence(regime_log: list) -> str:
    """Filters out -1 (warmup phases) and compresses consecutive identical regimes."""
    seq = []
    prev = None
    for r in regime_log:
        if r != -1:
            if prev == -1 or prev is None:
                seq.append(r)
        prev = r
    return " -> ".join(map(str, seq)) if seq else "None"

def calculate_delays(true_cps: list, detected_cps: list) -> list:
    """
    Calculates the delay for each detected change point.
    Matches each detection to the most recent true change point that occurred before it.
    """
    delays = []
    for d_cp in detected_cps:
        valid_true_cps = [t_cp for t_cp in true_cps if t_cp <= d_cp]
        if valid_true_cps:
            closest_true = max(valid_true_cps)
            delays.append(d_cp - closest_true)
    return delays

def generate_moment_matched_stream(tier="T1", segment_length=700, seed=42):
    """Generates the stream for Experiment 5.2: Robustness under matched moments."""
    np.random.seed(seed)
    sequence_ids = [1, 2, 3, 4, 2, 1, 4, 3, 1, 4, 2, 3] 
    
    if tier == "T1": 
        means, stds = {1: 0.0, 2: 3.0, 3: -3.0, 4: 6.0}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    elif tier == "T2": 
        means, stds = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, {1: 1.0, 2: 2.0, 3: 0.5, 4: 3.0}
    elif tier == "T3": 
        means, stds = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    else:
        raise ValueError("Tier must be 'T1', 'T2', or 'T3'")
        
    stream, actual_cps, current_len = [], [], 0
    
    for idx, dist_id in enumerate(sequence_ids):
        target_mean, target_std = means[dist_id], stds[dist_id]
        
        if dist_id == 1: 
            data = np.random.normal(target_mean, target_std, segment_length)
        elif dist_id == 2: 
            scale = target_std / np.sqrt(2)
            data = np.random.laplace(target_mean, scale, segment_length)
        elif dist_id == 3: 
            data = np.random.exponential(scale=target_std, size=segment_length)
            data = data - target_std + target_mean 
        elif dist_id == 4: 
            half_width = target_std * np.sqrt(3)
            data = np.random.uniform(target_mean - half_width, target_mean + half_width, segment_length)
            
        stream.extend(data)
        current_len += segment_length
        if idx < len(sequence_ids) - 1:
            actual_cps.append(current_len)
            
    return np.array(stream), actual_cps, sequence_ids

def plot_model_group(data, actual_cps, models_data, title, window_id):
    """Plots a subset of models in a distinct Matplotlib window to prevent visual crowding."""
    fig, axes = plt.subplots(len(models_data) + 1, 1, figsize=(12, 4 + 3 * len(models_data)), sharex=True)
    fig.canvas.manager.set_window_title(f"Simulation Window {window_id}")
    
    # Input Signal
    axes[0].plot(data, color="black", linewidth=0.5, alpha=0.8, label="Stream Data")
    for i, cp in enumerate(actual_cps):
        label = "True Change" if i == 0 else ""
        axes[0].axvline(x=cp, color="blue", linestyle="--", linewidth=1.5, label=label)
    
    axes[0].set_title(title)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Model Score Panels
    for ax, (scores, model, name, color, thresh_attr) in zip(axes[1:], models_data):
        ax.plot(scores, color=color, linewidth=1.2, label=f"{name} Score")
        
        thresh = getattr(model, thresh_attr, 0.1)
        ax.axhline(y=thresh, color="orange", linestyle="--", label=rf"Threshold = {thresh}")
        
        for i, cp in enumerate(model.global_changepoints):
            label = "Detection" if i == 0 else ""
            ax.axvline(x=cp, color="red", linestyle="-", linewidth=1.5, label=label)
            
        ax.set_title(f"{name} OWR-CPD Model")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step (t)")
    plt.tight_layout()

def main():
    # ==========================================
    # EXPERIMENT SETUP
    # ==========================================
    TIER = "T3"  
    
    configs = {
        "T1": {"n": 150, "m": 40,  "mmd_nu": 0.63, "seg_len": 700},
        "T2": {"n": 300, "m": 100, "mmd_nu": 0.99, "seg_len": 1000},
        "T3": {"n": 600, "m": 200, "mmd_nu": 1.00, "seg_len": 2000},
    }
    
    n, m = configs[TIER]["n"], configs[TIER]["m"]
    k_conf = 6 
    
    gamma = 0.015 
    dynamic_gamma = True # Active across all models
    
    print(f"Generating Open-World stream for Tier {TIER}...")
    data, actual_cps, true_seq = generate_moment_matched_stream(tier=TIER, segment_length=configs[TIER]["seg_len"])
    total_length = len(data)
    
    # Initialize Models 
    mmd_model = OWR_CPD(n=n, m=m, eta=0.06, nu=configs[TIER]["mmd_nu"], k_conf=k_conf, gamma=gamma)
    
    pearson_model = PearsonOWRCPD(n=n, m=m, tau=0.60, nu=0.48, k_conf=k_conf, reg_lambda=0.1, gamma=gamma, factor=1.0, dynamic_gamma=dynamic_gamma, normalize_distance=False)
    hybrid_model = HybridOWRCPD(n=n, m=m, tau=0.60, nu=0.90, k_conf=k_conf, reg_lambda=1.0, gamma=gamma, factor=1.0, dynamic_gamma=dynamic_gamma, normalize_distance=False)
    mahalanobis_model = MahalanobisOWRCPD(n=n, m=m, tau=0.60, nu=0.08, k_conf=k_conf, reg_lambda=0.1, gamma=gamma, factor=1.0, dynamic_gamma=dynamic_gamma, normalize_distance=False)
    
    dict_model = DictionaryOWRCPD(n=n, m=m, tau=0.20, nu=0.08, k_conf=k_conf, gamma=gamma, dict_max_size=n, coherence_thresh=0.95, ald_thresh=1e-3, dynamic_gamma=dynamic_gamma)

    # Run Simulation
    print(f"Starting concurrent monitoring (Total steps: {total_length})...")
    mmd_scores, pearson_scores, hybrid_scores, mah_scores, dict_scores = [], [], [], [], []
    
    for t in range(total_length):
        mmd_scores.append(mmd_model.update(data[t]))
        pearson_scores.append(pearson_model.update(data[t]))
        hybrid_scores.append(hybrid_model.update(data[t]))
        mah_scores.append(mahalanobis_model.update(data[t]))
        dict_scores.append(dict_model.update(data[t]))

    # Report
    print("\n" + "="*50)
    print(f"  SIMULATION SUMMARY REPORT: TIER {TIER}")
    print("="*50)
    print(f"\n GROUND TRUTH Sequence: {' -> '.join(map(str, true_seq))}")
    print(f" True Change Points   : {actual_cps}")

    models_summary = [
        ("MMD OWR MODEL", mmd_model),
        ("DICTIONARY OWR MODEL", dict_model),
        ("PEARSON OWR MODEL", pearson_model),
        ("HYBRID OWR MODEL", hybrid_model),
        ("MAHALANOBIS OWR MODEL", mahalanobis_model)
    ]

    for name, model in models_summary:
        cps = getattr(model, 'global_changepoints', [])
        regime_log = getattr(model, 'regime_log', [])
        
        delays = calculate_delays(actual_cps, cps)
        seq_str = get_regime_sequence(regime_log)
        
        if hasattr(model, 'regimes') and isinstance(model.regimes, list):
            total_regimes = len(model.regimes)
        else:
            total_regimes = len(set(r for r in regime_log if r != -1))
        
        print(f"\n {name}")
        print(f"  Detected Changes : {cps}")
        print(f"  Detection Delays : {delays}")
        print(f"  Mean Delay       : {np.mean(delays) if delays else 0.0:.2f} steps")
        print(f"  Total Regimes    : {total_regimes}")
        
        if name == "DICTIONARY OWR MODEL" and getattr(model, 'regimes', None):
            dict_sizes = [len(d) for d in model.regimes]
            if dict_sizes:
                print(f"  Mean Atoms/Dict  : {np.mean(dict_sizes):.1f}")
                print(f"  Max Atoms/Dict   : {np.max(dict_sizes)}")
            else:
                print(f"  Mean Atoms/Dict  : 0.0")
                print(f"  Max Atoms/Dict   : 0")
                
        print(f"  Regime Sequence  : {seq_str}")
    print("="*50)

    # Visualization
    group_1 = [
        (mmd_scores, mmd_model, "MMD", "teal", "eta"),
        (dict_scores, dict_model, "Dictionary", "dodgerblue", "tau")
    ]
    plot_model_group(data, actual_cps, group_1, f"Geometric & Sparse OWR-CPD Models (Tier {TIER})", window_id=1)

    group_2 = [
        (pearson_scores, pearson_model, "Pearson", "purple", "tau"),
        (hybrid_scores, hybrid_model, "Hybrid", "crimson", "tau"),
        (mah_scores, mahalanobis_model, "Mahalanobis", "darkmagenta", "tau")
    ]
    plot_model_group(data, actual_cps, group_2, f"Density-Ratio OWR-CPD Models (Tier {TIER})", window_id=2)

    plt.show()

if __name__ == "__main__":
    main()