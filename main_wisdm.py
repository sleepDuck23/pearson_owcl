import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import your models
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import Pearson_OWR_CPD
from model.owcl_p_coh import Hybrid_OWR_CPD
from model.owcl_mah import Mahalanobis_OWR_CPD

def load_wisdm_data():
    """
    Data Hook for WISDM. 
    Simulating 7 distinct physical activities (WISDM classes: Walk, Jog, Stairs, Sit, Stand, etc.)
    """
    print("Loading WISDM dataset...")
    block_size = 300
    np.random.seed(42)
    
    activities = {
        0: np.random.normal(0, 0.5, (block_size, 3)) + [1.0, 0.0, 9.8],   # Act 0
        1: np.random.normal(0, 1.5, (block_size, 3)) + [0.0, 2.0, 9.8],   # Act 1
        2: np.random.normal(0, 0.8, (block_size, 3)) + [0.5, 0.5, 9.8],   # Act 2
        3: np.random.normal(0, 0.7, (block_size, 3)) + [-0.5, -0.5, 9.8], # Act 3
        4: np.random.normal(0, 0.1, (block_size, 3)) + [0.0, 0.0, 9.8],   # Act 4 (Static)
        5: np.random.normal(0, 0.1, (block_size, 3)) + [0.0, 9.8, 0.0],   # Act 5 (Static)
        6: np.random.normal(0, 0.1, (block_size, 3)) + [9.8, 0.0, 0.0]    # Act 6 (Static)
    }
    return activities, block_size

def build_stream(sequence, activities, block_size):
    """Concatenates the activity blocks according to the chosen sequence."""
    stream = []
    true_cps = []
    current_step = 0
    
    for act_id in sequence:
        stream.append(activities[act_id])
        current_step += block_size
        true_cps.append(current_step)
        
    true_cps.pop() # Remove final boundary
    return np.vstack(stream), true_cps

def run_experiment():
    # --- CONFIGURATION ---
    USE_PCA = False
    PCA_D = 2  
    
    n = 150
    m = 50
    k_conf = 6
    
    sequence = [0, 1, 0, 2, 3, 2, 1, 3, 0, 2, 3]
    
    print(f"Building stream for sequence: {sequence}")
    activities, block_size = load_wisdm_data()
    stream, true_cps = build_stream(sequence, activities, block_size)
    
    # Preprocessing: StandardScaler is mandatory to center physical gravity biases
    scaler = StandardScaler()
    stream_processed = scaler.fit_transform(stream)
    
    if USE_PCA:
        print(f"Applying PCA (d={PCA_D})...")
        pca = PCA(n_components=PCA_D)
        stream_processed = pca.fit_transform(stream_processed)
    else:
        print("Skipping PCA. Streaming 3D signal directly...")

    # Initialize Models (Increased reg_lambda to 10.0 for raw 3D sensor noise)
    mmd_model = OWR_CPD(
        n=n, m=m, eta=0.06, nu=0.96, k_conf=k_conf
    )
    pearson_model = Pearson_OWR_CPD(
        n=n, m=m, tau=0.85, nu=0.71, k_conf=k_conf, reg_lambda=5.0, factor=2.5
    )
    hybrid_model = Hybrid_OWR_CPD(
        n=n, m=m, tau=0.85, nu=0.96, k_conf=k_conf, reg_lambda=5.0, factor=2.5
    )
    mahalanobis_model = Mahalanobis_OWR_CPD(
        n=n, m=m, tau=0.85, nu=0.025, k_conf=k_conf, reg_lambda=5.0, factor=2.5
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

    # Safely extract scores regardless of what the variable is named inside the class
    mmd_scores = getattr(mmd_model, 'mmd_scores', getattr(mmd_model, 'scores', []))
    pearson_scores = getattr(pearson_model, 'pearson_div_scores', getattr(pearson_model, 'scores', []))
    hybrid_scores = getattr(hybrid_model, 'pearson_div_scores', getattr(hybrid_model, 'scores', []))
    mah_scores = getattr(mahalanobis_model, 'pearson_div_scores', getattr(mahalanobis_model, 'scores', []))

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"WISDM Activity Stream (PCA={USE_PCA})", fontsize=14)

    models_data = [
        ("MMD Model", mmd_scores, mmd_model.eta, mmd_model.global_changepoints, 'teal'),
        ("Pearson Model", pearson_scores, pearson_model.tau, pearson_model.global_changepoints, 'purple'),
        ("Hybrid Model", hybrid_model.hybrid_scores, hybrid_model.tau, hybrid_model.global_changepoints, 'crimson'),
        ("Mahalanobis Model", mah_scores, mahalanobis_model.tau, mahalanobis_model.global_changepoints, 'indigo')
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
    plt.show()

    # --- TERMINAL SUMMARY OUTPUT ---
    def get_regime_sequence(regime_log):
        """Filters out -1 (warmup phases) and compresses consecutive identical regimes."""
        seq = []
        for r in regime_log:
            if r != -1:
                if not seq or seq[-1] != r:
                    seq.append(r)
        return seq

    print("\n" + "="*50)
    print(f" WISDM EXPERIMENT RESULTS (PCA={USE_PCA})")
    print("="*50)
    print(f" GROUND TRUTH Sequence: {' -> '.join(map(str, sequence))}")
    print(f" True Change Points   : {true_cps}\n")

    models = [
        ("MMD OWR MODEL", mmd_model),
        ("PEARSON OWR MODEL", pearson_model),
        ("HYBRID OWR MODEL", hybrid_model),
        ("MAHALANOBIS OWR MODEL", mahalanobis_model)
    ]

    for name, model in models:
        # Safely extract logs
        cps = getattr(model, 'global_changepoints', [])
        regime_log = getattr(model, 'regime_log', [])
        seq = get_regime_sequence(regime_log)
        
        print(f" {name}")
        print(f"  Detected Changes : {cps}")
        print(f"  Total Regimes    : {len(set(seq))}")
        print(f"  Regime Sequence  : {' -> '.join(map(str, seq))}\n")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_experiment()