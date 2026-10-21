import numpy as np
from main_mmd_pearson import (
    detection_indices, get_regime_sequence, calculate_delays, print_detection_summary,
)
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Import your models
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD       
from model.owcl_p_coh import HybridOWRCPD          
from model.owcl_mah import MahalanobisOWRCPD       



def generate_mnist_stream(sequence, samples_per_digit=300):
    """
    Downloads MNIST and builds a sequential stream based on the provided digit sequence.
    """
    print("Loading MNIST dataset (this may take a moment)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0 
    y = y.astype(int)
    
    if not sequence or samples_per_digit < 1:
        raise ValueError("Provide a nonempty sequence and positive segment length")
    rng = np.random.RandomState(42)
    pools = {}
    offsets = {}
    for digit in sorted(set(sequence)):
        indices = np.flatnonzero(y == digit)
        needed = sequence.count(digit) * samples_per_digit
        if len(indices) < needed:
            raise ValueError(f"Digit {digit} needs {needed} distinct images; only {len(indices)} available")
        pools[digit] = rng.permutation(indices)
        offsets[digit] = 0
    stream = []
    actual_cps = []
    current_len = 0
    
    print(f"Building stream for sequence: {sequence}")
    for idx, digit in enumerate(sequence):
        start = offsets[digit]
        selected_indices = pools[digit][start:start + samples_per_digit]
        offsets[digit] += samples_per_digit
        segment = X[selected_indices]

        stream.extend(segment)
        current_len += samples_per_digit
        
        if idx < len(sequence) - 1 and digit != sequence[idx + 1]:
            actual_cps.append(current_len)
            
    return np.array(stream), actual_cps

def main():
    # ==========================================
    # EXPERIMENT SETUP
    # ==========================================
    # The paper tests two specific sequences:
    SEQ_A = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5] #[0,1,2,1,3,4,5,6,3,0,3]
    SEQ_B = [4, 9, 4, 8, 3, 8, 9, 3, 4, 8, 3] #[0,1,0,2,3,2,1,3,0,2,3]
    
    TARGET_SEQ = SEQ_B 
    USE_PCA = True  # Toggle this to False to test raw 784D space
    PCA_DIM = 50
    
    # Paper parameters (Section 5.3)
    n = 150
    m = 40
    k_conf = 6
    gamma = 1.0            # RBF kernel bandwidth
    dynamic_gamma = True   # Use dynamic gamma adjustment
    normalize_distance = True  
    
    # Generate the raw stream
    raw_data, actual_cps = generate_mnist_stream(TARGET_SEQ, samples_per_digit=300)
    
    if USE_PCA:
        print(f"Applying PCA to reduce dimensions to d={PCA_DIM}...")
        pca = PCA(n_components=PCA_DIM, random_state=42)
        # Fit on the initial reference window only, before monitoring starts.
        pca.fit(raw_data[:n])
        data = pca.transform(raw_data)
    else:
        print("Using raw 784-dimensional pixel space...")
        data = raw_data

    total_length = len(data)
    
    # ==========================================
    # INITIALIZE MODELS
    # ==========================================
    mmd_model = OWR_CPD(
        n=n, 
        m=m, 
        eta=0.06,   # Paper calibrated detection threshold
        nu=0.96,    # Paper calibrated novelty threshold
        k_conf=k_conf,
        gamma=gamma,
        dynamic_gamma=dynamic_gamma
    )
    
    pearson_model = PearsonOWRCPD(
        n=n, 
        m=m, 
        tau=0.46, 
        nu=0.71, 
        k_conf=k_conf, 
        reg_lambda=10,
        gamma=gamma,
        factor=1.0,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=normalize_distance
    )

    hybrid_model = HybridOWRCPD(
        n=n, 
        m=m, 
        tau=0.38, 
        nu=0.96, 
        k_conf=k_conf, 
        reg_lambda=10,
        gamma=gamma,
        factor=0.75,  
        dynamic_gamma=dynamic_gamma,
        normalize_distance=normalize_distance
    )

    mahalanobis_model = MahalanobisOWRCPD(
        n=n, 
        m=m, 
        tau=0.30, 
        nu=0.025, 
        k_conf=k_conf, 
        reg_lambda=10,
        gamma=gamma,
        factor=0.5,
        dynamic_gamma=dynamic_gamma,
        normalize_distance=normalize_distance
    )

    # ==========================================
    # RUN STREAMING SIMULATION
    # ==========================================
    print(f"Starting concurrent monitoring (Total steps: {total_length})...")
    mmd_scores, pearson_scores, hybrid_scores, mah_scores = [], [], [], []
    
    for t in range(total_length):
        mmd_scores.append(mmd_model.update(data[t]))
        pearson_scores.append(pearson_model.update(data[t]))
        hybrid_scores.append(hybrid_model.update(data[t]))
        mah_scores.append(mahalanobis_model.update(data[t]))

    # ==========================================
    # REPORTING & VISUALIZATION
    # ==========================================
    print("\n" + "="*50)
    print(f"  MNIST EXPERIMENT (PCA={USE_PCA})")
    print("="*50)
    print(f"\n GROUND TRUTH Sequence: {' -> '.join(map(str, TARGET_SEQ))}")
    print(f" True Change Points   : {actual_cps}")

    models = [
        ("MMD", mmd_model, mmd_scores, "teal"),
        ("PEARSON", pearson_model, pearson_scores, "purple"),
        ("HYBRID", hybrid_model, hybrid_scores, "crimson"),
        ("MAHALANOBIS", mahalanobis_model, mah_scores, "darkmagenta")
    ]

    for name, model, scores, color in models:
        print(f"\n {name} OWR MODEL")
        print_detection_summary(actual_cps, model)
    print("="*50)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"MNIST Digit Stream (PCA={USE_PCA})", fontsize=14)

    for ax, (name, model, scores, color) in zip(axes, models):
        ax.plot(scores, color=color, linewidth=1.2, label=f"{name} Score")
        thresh = getattr(model, 'eta', getattr(model, 'tau', 0.1))
        ax.axhline(y=thresh, color="orange", linestyle="--", label="Detection Threshold")
        
        for cp in actual_cps:
            ax.axvline(x=cp, color="black", linestyle=":", alpha=0.5, linewidth=1.5)
        for cp in detection_indices(model):
            ax.axvline(x=cp, color="red", linestyle="-", linewidth=1.5)
            
        ax.set_title(f"{name} OWR-CPD Model")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
