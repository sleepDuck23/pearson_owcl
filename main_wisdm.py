import numpy as np
from main_mmd_pearson import (
    detection_indices, get_regime_sequence, calculate_delays, print_detection_summary,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import your models
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD       # Refactored class
from model.owcl_p_coh import HybridOWRCPD          # Refactored class
from model.owcl_mah import MahalanobisOWRCPD       # Refactored class



def generate_synthetic_activity_data(blocks_per_activity=None):
    """Synthetic Gaussian activity surrogate; this does not load WISDM recordings."""
    print("Generating synthetic activity data (WISDM placeholder)...")
    block_size = 2500
    rng = np.random.RandomState(42)
    specifications = {
        0: (0.5, [1.0, 0.0, 9.8]),
        1: (1.5, [0.0, 2.0, 9.8]),
        2: (0.8, [0.5, 0.5, 9.8]),
        3: (0.7, [-0.5, -0.5, 9.8]),
        4: (0.1, [0.0, 0.0, 9.8]),
        5: (0.1, [0.0, 9.8, 0.0]),
        6: (0.1, [9.8, 0.0, 0.0]),
    }
    counts = blocks_per_activity if blocks_per_activity is not None else dict.fromkeys(specifications, 1)
    activities = {}
    for act, count in sorted(counts.items()):
        if act not in specifications or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("Activity IDs and block counts must be valid")
        std, mean = specifications[act]
        activities[act] = rng.normal(0, std, (count * block_size, 3)) + mean
    return activities, block_size


def build_stream(sequence, activities, block_size):
    """Consume disjoint blocks per activity and derive boundaries from their lengths."""
    if not sequence or block_size < 1:
        raise ValueError("Provide a nonempty sequence and positive block size")
    stream, true_cps = [], []
    offsets = {act: 0 for act in sequence}
    current_step = 0
    for i, act_id in enumerate(sequence):
        if act_id not in activities:
            raise ValueError(f"No observations available for activity {act_id}")
        start = offsets[act_id]
        block = np.asarray(activities[act_id][start:start + block_size])
        if len(block) != block_size:
            raise ValueError(f"Not enough distinct observations for activity {act_id}")
        if block.ndim != 2 or not np.isfinite(block).all():
            raise ValueError(f"Activity {act_id} must contain finite observation rows")
        if i and act_id != sequence[i - 1]:
            true_cps.append(current_step)
        stream.append(block)
        offsets[act_id] += len(block)
        current_step += len(block)
    return np.vstack(stream), true_cps

def run_experiment():
    # --- CONFIGURATION ---
    USE_PCA = False
    PCA_D = 2  
    
    n = 150
    m = 50
    k_conf = 6
    DOWNSAMPLE_RATE = 5
    dynamic_gamma = True
    
    sequence = [0, 1, 0, 2, 3, 2, 1, 3, 0, 2, 3]
    
    print(f"Building stream for sequence: {sequence}")
    counts = {act: sequence.count(act) for act in set(sequence)}
    activities, block_size = generate_synthetic_activity_data(counts)
    stream, true_cps = build_stream(sequence, activities, block_size)

    # Downsample before fitting: n always denotes observations seen by the model.
    print(f"Downsampling stream by factor of {DOWNSAMPLE_RATE}...")
    stream = stream[::DOWNSAMPLE_RATE]
    true_cps = [(cp + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE for cp in true_cps]
    
    # Freeze preprocessing after the initial reference window; no future fitting.
    scaler = StandardScaler()
    scaler.fit(stream[:n])
    stream_processed = scaler.transform(stream)
    
    if USE_PCA:
        print(f"Applying PCA (d={PCA_D})...")
        pca = PCA(n_components=PCA_D)
        pca.fit(stream_processed[:n])
        stream_processed = pca.transform(stream_processed)
    else:
        print("Skipping PCA. Streaming 3D signal directly...")

    # Initialize Models using Refactored Names
    mmd_model = OWR_CPD(
        n=n, 
        m=m, 
        eta=0.06, 
        nu=0.96, 
        k_conf=k_conf,
        dynamic_gamma=dynamic_gamma
    )
    
    pearson_model = PearsonOWRCPD(
        n=n, 
        m=m, 
        tau=0.40, 
        nu=0.71, 
        k_conf=k_conf, 
        reg_lambda=1e-2, 
        factor=1.0,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )
    
    hybrid_model = HybridOWRCPD(
        n=n, 
        m=m, 
        tau=0.50, 
        nu=0.96, 
        k_conf=k_conf, 
        reg_lambda=1e-2, 
        factor=1.0,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )
    
    mahalanobis_model = MahalanobisOWRCPD(
        n=n, 
        m=m, 
        tau=0.50, 
        nu=0.025, 
        k_conf=k_conf, 
        reg_lambda=1e-2, 
        factor=1.0,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=False
    )

    total_steps = len(stream_processed)
    print(f"Starting concurrent monitoring (Total steps: {total_steps})...")

    # Run Stream
    for t in range(total_steps):
        x = stream_processed[t]
        mmd_model.update(x)
        pearson_model.update(x)
        hybrid_model.update(x)
        mahalanobis_model.update(x)

    # Safely extract scores
    mmd_scores = getattr(mmd_model, 'mmd_scores', getattr(mmd_model, 'scores', []))
    pearson_scores = getattr(pearson_model, 'pearson_div_scores', getattr(pearson_model, 'scores', []))
    hybrid_scores = getattr(hybrid_model, 'hybrid_scores', getattr(hybrid_model, 'scores', []))
    mah_scores = getattr(mahalanobis_model, 'pearson_div_scores', getattr(mahalanobis_model, 'scores', []))

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Synthetic Activity Stream (WISDM placeholder) (PCA={USE_PCA})", fontsize=14)

    models_data = [
        ("MMD Model", mmd_scores, mmd_model.eta, detection_indices(mmd_model), 'teal'),
        ("Pearson Model", pearson_scores, pearson_model.tau, detection_indices(pearson_model), 'purple'),
        ("Hybrid Model", hybrid_scores, hybrid_model.tau, detection_indices(hybrid_model), 'crimson'),
        ("Mahalanobis Model", mah_scores, mahalanobis_model.tau, detection_indices(mahalanobis_model), 'indigo')
    ]

    for idx, (name, scores, threshold, changepoints, color) in enumerate(models_data):
        ax = axes[idx]
        if len(scores) > 0:
            ax.plot(scores, label=f"{name} Score", color=color, linewidth=1.2)
            ax.axhline(y=threshold, color='orange', linestyle='--', label="Detection Threshold")
            
            for cp in true_cps:
                ax.axvline(x=cp, color='black', linestyle=':', alpha=0.6)
            for cp in changepoints:
                ax.axvline(x=cp, color='red', linestyle='-', alpha=0.8)
                
        ax.set_title(name, fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # --- TERMINAL SUMMARY OUTPUT ---
    print("\n" + "="*50)
    print(f" SYNTHETIC ACTIVITY RESULTS (WISDM placeholder) (PCA={USE_PCA})")
    print("="*50)
    print(f" GROUND TRUTH Sequence: {' -> '.join(map(str, sequence))}")
    print(f" True Change Points   : {true_cps}\n")

    models_summary = [
        ("MMD OWR MODEL", mmd_model),
        ("PEARSON OWR MODEL", pearson_model),
        ("HYBRID OWR MODEL", hybrid_model),
        ("MAHALANOBIS OWR MODEL", mahalanobis_model)
    ]

    for name, model in models_summary:
        print(f" {name}")
        print_detection_summary(true_cps, model)
        print()
    print("="*50 + "\n")

    plt.show()

if __name__ == "__main__":
    run_experiment()
