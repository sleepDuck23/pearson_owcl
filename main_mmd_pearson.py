import numpy as np
import matplotlib.pyplot as plt

# Import the data generator and all four OWR models
from data.generation import generate_regime_stream
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import Pearson_OWR_CPD
from model.owcl_p_coh import Hybrid_OWR_CPD
from model.owcl_mah import Mahalanobis_OWR_CPD

def get_regime_sequence(regime_log):
    """
    Extracts the sequence of assigned regimes. 
    Registers a new assignment every time the model exits a transition state (-1),
    allowing it to correctly show sequential identical regimes (e.g., 0 -> 1 -> 1).
    """
    seq = []
    prev = None
    for r in regime_log:
        if r != -1:
            # If this is the first item, or if we just exited a transition phase
            if prev == -1 or prev is None:
                seq.append(r)
        prev = r
    return " -> ".join(map(str, seq)) if seq else "None"

def main():
    # 1. Generate the stream data (Regimes: A -> B -> C -> A)
    total_length = 6000
    true_cps = [1000, 2000, 3000, 4000, 5000]  # True change points
    n = 150  # ref Window size
    m = 50  # test Window size
    gamma = 0.01  # RBF kernel bandwidth
    k_conf = 3  # Consecutive windows to confirm change
    
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
        gamma=gamma
    )
    
    pearson_model = Pearson_OWR_CPD(
        n=n, 
        m=m, 
        tau=0.1,  
        nu=0.90,  
        k_conf=k_conf, 
        reg_lambda=1e-3, 
        gamma=gamma
    )

    hybrid_model = Hybrid_OWR_CPD(
        n=n, 
        m=m, 
        tau=0.1,  
        nu=0.90,  
        k_conf=k_conf, 
        reg_lambda=1e-5, 
        gamma=gamma
    )

    mahalanobis_model = Mahalanobis_OWR_CPD(
        n=n, 
        m=m, 
        tau=0.1,  
        nu=0.08,  
        k_conf=k_conf, 
        reg_lambda=1e-5, 
        gamma=gamma
    )

    # 3. Run the online simulation
    print("Starting concurrent monitoring...")
    mmd_scores = []
    pearson_scores = []
    hybrid_scores = []
    mah_scores = []
    
    for t in range(total_length):
        score_mmd = mmd_model.update(data[t])
        score_pearson = pearson_model.update(data[t])
        score_hybrid = hybrid_model.update(data[t])
        score_mah = mahalanobis_model.update(data[t])
        
        mmd_scores.append(score_mmd)
        pearson_scores.append(score_pearson)
        hybrid_scores.append(score_hybrid)
        mah_scores.append(score_mah)

    # 4. Readable Summary Report
    print("\n" + "="*50)
    print("  SIMULATION SUMMARY REPORT")
    print("="*50)

    print("\n GROUND TRUTH")
    print(f"  Change Points      : {actual_cps}")
    print(f"  Regime Sequence    : A -> B -> C -> B -> A -> D")

    print("\n MMD OWR MODEL")
    print(f"  Detected Changes   : {mmd_model.global_changepoints}")
    print(f"  Total Regimes      : {len(mmd_model.regimes)}")
    print(f"  Regime Sequence    : {get_regime_sequence(mmd_model.regime_log)}")

    print("\n PEARSON OWR MODEL")
    print(f"  Detected Changes   : {pearson_model.global_changepoints}")
    print(f"  Total Regimes      : {len(pearson_model.regimes)}")
    print(f"  Regime Sequence    : {get_regime_sequence(pearson_model.regime_log)}")

    print("\n HYBRID OWR MODEL (Pearson CPD + MMD Coherence)")
    print(f"  Detected Changes   : {hybrid_model.global_changepoints}")
    print(f"  Total Regimes      : {len(hybrid_model.regimes)}")
    print(f"  Regime Sequence    : {get_regime_sequence(hybrid_model.regime_log)}")
    
    print("\n MAHALANOBIS OWR MODEL (Pearson CPD + Mahalanobis Dist)")
    print(f"  Detected Changes   : {mahalanobis_model.global_changepoints}")
    print(f"  Total Regimes      : {len(mahalanobis_model.regimes)}")
    print(f"  Regime Sequence    : {get_regime_sequence(mahalanobis_model.regime_log)}")
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
    ax2.axhline(y=mmd_model.eta, color="orange", linestyle="--", linewidth=2, label=f"Threshold $\eta$ = {mmd_model.eta}")
    for i, cp in enumerate(mmd_model.global_changepoints):
        label = "MMD Detection" if i == 0 else ""
        ax2.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
    ax2.set_title("OWR-CPD (MMD) Model")
    ax2.set_ylabel("MMD Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Pearson Model Score ---
    ax3.plot(pearson_scores, color="purple", linewidth=1.5, label="Pearson Divergence")
    ax3.axhline(y=pearson_model.tau, color="green", linestyle="--", linewidth=2, label=f"Threshold $\\tau$ = {pearson_model.tau}")
    for i, cp in enumerate(pearson_model.global_changepoints):
        label = "Pearson Detection" if i == 0 else ""
        ax3.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
    ax3.set_title("Pearson OWR-CPD Model")
    ax3.set_ylabel("Pearson Score")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Hybrid Model Score ---
    ax4.plot(hybrid_scores, color="crimson", linewidth=1.5, label="Hybrid Score (Pearson CPD)")
    ax4.axhline(y=hybrid_model.tau, color="darkorange", linestyle="--", linewidth=2, label=f"Threshold $\\tau$ = {hybrid_model.tau}")
    for i, cp in enumerate(hybrid_model.global_changepoints):
        label = "Hybrid Detection" if i == 0 else ""
        ax4.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)
    ax4.set_title("Hybrid OWR-CPD Model")
    ax4.set_ylabel("Hybrid Score")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Mahalanobis Model Score ---
    ax5.plot(mah_scores, color="darkmagenta", linewidth=1.5, label="Mahalanobis Score (Pearson CPD)")
    ax5.axhline(y=mahalanobis_model.tau, color="goldenrod", linestyle="--", linewidth=2, label=f"Threshold $\\tau$ = {mahalanobis_model.tau}")
    for i, cp in enumerate(mahalanobis_model.global_changepoints):
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