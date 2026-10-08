import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import the new PAMAP2 data manager
from data.manage_pamap2 import load_pamap2_blocks, load_pamap2_full_blocks

# Import your models
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import Pearson_OWR_CPD
from model.owcl_p_coh import Hybrid_OWR_CPD
from model.owcl_mah import Mahalanobis_OWR_CPD

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
    PCA_D = 5  
    
    # PAMAP2 is sampled at 100Hz (vs WISDM's 20Hz). 
    # To capture the same physical time duration, we must increase the window sizes.
    # n=750 is 7.5 seconds of physical movement at 100Hz.
    n = 500
    m = 100
    k_conf = 6
    
    sequence = [0, 1, 0, 2, 3, 2, 1, 3, 0, 2, 3]
    
    print(f"Building PAMAP2 stream for sequence: {sequence}")
    
    # Fetch data directly from our dedicated manager
    #activities, block_size = load_pamap2_full_blocks(block_size=1500) # 15 seconds per block
    activities, block_size = load_pamap2_blocks(block_size=1500) # 15 seconds per block
    stream, true_cps = build_stream(sequence, activities, block_size=1500)
    
    # Preprocessing
    scaler = StandardScaler()
    stream_processed = scaler.fit_transform(stream)
    
    if USE_PCA:
        print(f"Applying PCA (d={PCA_D})...")
        pca = PCA(n_components=PCA_D)
        stream_processed = pca.fit_transform(stream_processed)
    else:
        print("Skipping PCA. Streaming 3D signal directly...")

    # Initialize Models
    mmd_model = OWR_CPD(n=n, m=m, eta=0.75, nu=0.90, k_conf=k_conf, factor=1.0)
    pearson_model = Pearson_OWR_CPD(n=n, m=m, tau=0.70, nu=0.71, k_conf=k_conf, reg_lambda=0.075, factor=0.90)
    hybrid_model = Hybrid_OWR_CPD(n=n, m=m, tau=0.70, nu=0.96, k_conf=k_conf, reg_lambda=0.01, factor=0.90)
    mahalanobis_model = Mahalanobis_OWR_CPD(n=n, m=m, tau=0.70, nu=0.05, k_conf=k_conf, reg_lambda=0.05, factor=0.90)

    total_steps = len(stream_processed)
    print(f"Starting concurrent monitoring (Total steps: {total_steps})...")

    # Run Stream
    for t in range(total_steps):
        x = stream_processed[t]
        mmd_model.update(x)
        pearson_model.update(x)
        hybrid_model.update(x)
        mahalanobis_model.update(x)

    # Extract Scores
    mmd_scores = getattr(mmd_model, 'mmd_scores', getattr(mmd_model, 'scores', []))
    pearson_scores = getattr(pearson_model, 'pearson_div_scores', getattr(pearson_model, 'scores', []))
    hybrid_scores = getattr(hybrid_model, 'hybrid_scores', getattr(hybrid_model, 'scores', []))
    mah_scores = getattr(mahalanobis_model, 'pearson_div_scores', getattr(mahalanobis_model, 'scores', []))

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"PAMAP2 Activity Stream (PCA={USE_PCA})", fontsize=14)

    models_data = [
        ("MMD Model", mmd_scores, mmd_model.eta, mmd_model.global_changepoints, 'teal'),
        ("Pearson Model", pearson_scores, pearson_model.tau, pearson_model.global_changepoints, 'purple'),
        ("Hybrid Model", hybrid_scores, hybrid_model.tau, hybrid_model.global_changepoints, 'crimson'),
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
        seq = []
        for r in regime_log:
            if r != -1:
                if not seq or seq[-1] != r:
                    seq.append(r)
        return seq

    print("\n" + "="*50)
    print(f" PAMAP2 EXPERIMENT RESULTS (PCA={USE_PCA})")
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