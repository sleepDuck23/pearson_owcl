import numpy as np
import matplotlib.pyplot as plt

# Import the data generator and all OWR models
from data.generation import generate_regime_stream
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD       
from model.owcl_p_coh import HybridOWRCPD
from model.owcl_mah import MahalanobisOWRCPD       
from model.owcl_dict import DictionaryOWRCPD # Assumed class name for the new dict model

def get_regime_sequence(regime_log: list) -> str:
    """
    Extracts the sequence of assigned regimes. 
    Registers a new assignment every time the model exits a transition state (-1).
    """
    seq = []
    prev = None
    for r in regime_log:
        if r != -1:
            if prev == -1 or prev is None:
                seq.append(r)
        prev = r
    return " -> ".join(map(str, seq)) if seq else "None"

def calculate_delays(true_cps: list, detected_cps: list) -> list:
    """Calculates the delay for each detected change point."""
    delays = []
    for d_cp in detected_cps:
        valid_true_cps = [t_cp for t_cp in true_cps if t_cp <= d_cp]
        if valid_true_cps:
            closest_true = max(valid_true_cps)
            delays.append(d_cp - closest_true)
    return delays

def plot_model_group(data, actual_cps, models_data, title, window_id):
    """
    Helper function to plot a subset of models in a distinct window.
    models_data is a list of tuples: (scores, model_obj, name, color, thresh_attr)
    """
    fig, axes = plt.subplots(len(models_data) + 1, 1, figsize=(12, 4 + 3 * len(models_data)), sharex=True)
    fig.canvas.manager.set_window_title(f"Simulation Window {window_id}")
    
    # --- Panel 1: Input Signal ---
    axes[0].plot(data, color="black", linewidth=0.8, alpha=0.7, label="Stream Data")
    for i, cp in enumerate(actual_cps):
        label = "True Change" if i == 0 else ""
        axes[0].axvline(x=cp, color="blue", linestyle="--", linewidth=2, label=label)
    
    axes[0].set_title(title)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # --- Model Score Panels ---
    for ax, (scores, model, name, color, thresh_attr) in zip(axes[1:], models_data):
        ax.plot(scores, color=color, linewidth=1.5, label=f"{name} Score")
        
        # Get threshold (either eta or tau depending on the model)
        thresh = getattr(model, thresh_attr, 0.1)
        ax.axhline(y=thresh, color="orange", linestyle="--", linewidth=2, label=rf"Threshold = {thresh}")
        
        for i, cp in enumerate(model.global_changepoints):
            label = f"{name} Detection" if i == 0 else ""
            ax.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
            
        ax.set_title(f"{name} OWR-CPD Model")
        ax.set_ylabel("Score")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step (t)")
    plt.tight_layout()

def main():
    # 1. Generate the stream data (Regimes: A -> B -> C -> A)
    total_length = 6000
    true_cps = [1000, 2000, 3000, 4000, 5000] 
    n = 150        
    m = 50         
    gamma = 0.6    
    k_conf = 6     
    dynamic_gamma = True 
    
    print("Generating Open-World stream data...\n")
    data, actual_cps = generate_regime_stream(
        length=total_length, 
        changepoints=true_cps, 
        means=[0.0, 5.0, -4.0,  5.0, 0.0, 10.0],  
        stds=[1.0, 1.0, 0.5, 1.0, 1.0, 2.0],
        seed=42
    )

    # 2. Initialize Models
    mmd_model = OWR_CPD(n=n, m=m, eta=0.1, nu=0.90, k_conf=k_conf, gamma=gamma)
    
    pearson_model = PearsonOWRCPD(
        n=n, m=m, tau=0.60, nu=0.50, k_conf=k_conf, 
        reg_lambda=1e-5, gamma=gamma, dynamic_gamma=dynamic_gamma, normalize_distance=False
    )
    hybrid_model = HybridOWRCPD(
        n=n, m=m, tau=0.60, nu=0.90, k_conf=k_conf, 
        reg_lambda=1e-5, gamma=gamma, dynamic_gamma=dynamic_gamma, normalize_distance=False
    )
    mahalanobis_model = MahalanobisOWRCPD(
        n=n, m=m, tau=0.60, nu=0.08, k_conf=k_conf, 
        reg_lambda=1e-5, gamma=gamma, dynamic_gamma=dynamic_gamma, normalize_distance=False
    )
    # New Dictionary Model (Assuming standard OWR-CPD initialization + dict params)
    dict_model = DictionaryOWRCPD(
        n=n, m=m, 
        tau=0.35, nu=0.08, k_conf=k_conf, gamma=gamma,
        dict_max_size=50, coherence_thresh=0.95, ald_thresh=0.1
    )

    # 3. Run the online simulation
    print("Starting concurrent monitoring...")
    mmd_scores, pearson_scores, hybrid_scores, mah_scores, dict_scores = [], [], [], [], []
    
    for t in range(total_length):
        mmd_scores.append(mmd_model.update(data[t]))
        pearson_scores.append(pearson_model.update(data[t]))
        hybrid_scores.append(hybrid_model.update(data[t]))
        mah_scores.append(mahalanobis_model.update(data[t]))
        dict_scores.append(dict_model.update(data[t]))

    # 4. Readable Summary Report
    print("\n" + "="*50)
    print("  SIMULATION SUMMARY REPORT")
    print("="*50)
    print("\n GROUND TRUTH")
    print(f"  Change Points      : {actual_cps}")
    print(f"  Regime Sequence    : A -> B -> C -> B -> A -> D")

    models_list = [
        ("MMD", mmd_model),
        ("PEARSON", pearson_model),
        ("HYBRID", hybrid_model),
        ("MAHALANOBIS", mahalanobis_model),
        ("DICTIONARY", dict_model)
    ]

    for name, model in models_list:
        delays = calculate_delays(actual_cps, model.global_changepoints)
        print(f"\n {name} OWR MODEL")
        print(f"  Detected Changes   : {model.global_changepoints}")
        print(f"  Detection Delays   : {delays}")
        print(f"  Mean Delay         : {np.mean(delays) if delays else 0.0:.2f} steps")
        print(f"  Total Regimes      : {len(getattr(model, 'regimes', []))}")
        print(f"  Regime Sequence    : {get_regime_sequence(model.regime_log)}")
    print("\n" + "="*50)

    # 5. Visualization (Split into two separate windows to avoid squeezing)
    
    # Window 1: The Geometric/RKHS approaches
    group_1 = [
        (mmd_scores, mmd_model, "MMD", "teal", "eta"),
        (dict_scores, dict_model, "Dictionary", "dodgerblue", "tau")
    ]
    plot_model_group(data, actual_cps, group_1, "Geometric & Sparse OWR-CPD Models", window_id=1)

    # Window 2: The Pearson-based density ratio approaches
    group_2 = [
        (pearson_scores, pearson_model, "Pearson", "purple", "tau"),
        (hybrid_scores, hybrid_model, "Hybrid", "crimson", "tau"),
        (mah_scores, mahalanobis_model, "Mahalanobis", "darkmagenta", "tau")
    ]
    plot_model_group(data, actual_cps, group_2, "Density-Ratio OWR-CPD Models", window_id=2)

    plt.show()

if __name__ == "__main__":
    main()