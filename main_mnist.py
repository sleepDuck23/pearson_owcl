import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Import your models
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import Pearson_OWR_CPD
from model.owcl_p_coh import Hybrid_OWR_CPD
from model.owcl_mah import Mahalanobis_OWR_CPD

def get_regime_sequence(regime_log):
    """Extracts the sequence of assigned regimes."""
    seq = []
    prev = None
    for r in regime_log:
        if r != -1:
            if prev == -1 or prev is None:
                seq.append(r)
        prev = r
    return " -> ".join(map(str, seq)) if seq else "None"

def generate_mnist_stream(sequence, samples_per_digit=300):
    """
    Downloads MNIST and builds a sequential stream based on the provided digit sequence.
    """
    print("Loading MNIST dataset (this may take a moment)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0 
    y = y.astype(int)
    
    stream = []
    actual_cps = []
    current_len = 0
    
    print(f"Building stream for sequence: {sequence}")
    for idx, digit in enumerate(sequence):
        # Extract all images for the target digit
        digit_images = X[y == digit]
        
        # Randomly sample the required number of images
        np.random.seed(42 + idx) # Seed for reproducibility
        selected_indices = np.random.choice(len(digit_images), samples_per_digit, replace=False)
        segment = digit_images[selected_indices]
        
        stream.extend(segment)
        current_len += samples_per_digit
        
        if idx < len(sequence) - 1:
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
    
    # Generate the raw stream
    raw_data, actual_cps = generate_mnist_stream(TARGET_SEQ, samples_per_digit=300)
    
    if USE_PCA:
        print(f"Applying PCA to reduce dimensions to d={PCA_DIM}...")
        pca = PCA(n_components=PCA_DIM, random_state=42)
        data = pca.fit_transform(raw_data)
    else:
        print("Using raw 784-dimensional pixel space...")
        data = raw_data

    total_length = len(data)
    
    # ==========================================
    # INITIALIZE MODELS
    # ==========================================
    # Note: The paper provides exact thresholds for the MMD model.
    # You will need to tune tau and nu for the Pearson-based models.
    mmd_model = OWR_CPD(
        n=n, m=m, 
        eta=0.06,   # Paper calibrated detection threshold
        nu=0.96,    # Paper calibrated novelty threshold
        k_conf=k_conf
    )
    
    pearson_model = Pearson_OWR_CPD(
        n=n, m=m, tau=0.725, nu=0.71, k_conf=k_conf, reg_lambda=1e-0
    )

    hybrid_model = Hybrid_OWR_CPD(
        n=n, m=m, tau=0.725, nu=0.96, k_conf=k_conf, reg_lambda=1e-0
    )

    mahalanobis_model = Mahalanobis_OWR_CPD(
        n=n, m=m, tau=0.725, nu=0.025, k_conf=k_conf, reg_lambda=1e-0
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
        print(f"  Detected Changes : {model.global_changepoints}")
        print(f"  Total Regimes    : {len(model.regimes)}")
        print(f"  Regime Sequence  : {get_regime_sequence(model.regime_log)}")
    print("="*50)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"MNIST Digit Stream (PCA={USE_PCA})", fontsize=14)

    for ax, (name, model, scores, color) in zip(axes, models):
        ax.plot(scores, color=color, linewidth=1.2, label=f"{name} Score")
        thresh = getattr(model, 'eta', getattr(model, 'tau', 0.1))
        ax.axhline(y=thresh, color="orange", linestyle="--", label="Detection Threshold")
        
        for cp in actual_cps:
            ax.axvline(x=cp, color="black", linestyle=":", alpha=0.5, linewidth=1.5)
        for cp in model.global_changepoints:
            ax.axvline(x=cp, color="red", linestyle="-", linewidth=1.5)
            
        ax.set_title(f"{name} OWR-CPD Model")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()