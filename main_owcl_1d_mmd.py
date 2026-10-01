import numpy as np
import matplotlib.pyplot as plt

# Import the new data generator and the MMD model
from data.generation import generate_regime_stream
from model.owcl_mmd import OWR_CPD

def main():
    # 1. Generate the stream data (Recurrent Regime A -> B -> C -> A)
    total_length = 3000
    true_cps = [700, 1500, 2200]
    
    print("Generating open-world stream data...")
    data, actual_cps = generate_regime_stream(
        length=total_length, 
        changepoints=true_cps, 
        means=[0.0, 5.0, -3.0, 0.0],  
        stds=[1.0, 1.5, 0.5, 1.0],
        seed=42
    )

    # 2. Initialize the OWR-CPD (MMD) model
    # Parameters aligned with the ICASSP paper reference setup
    eta_threshold = 0.2  # Detection threshold (MMD)
    model = OWR_CPD(
        n=150,           # Reference window size
        m=40,            # Analysis window size
        eta=eta_threshold, 
        nu=0.90,         # Coherence novelty threshold
        k_conf=6,        # Confirmation persistence
        gamma=0.1        # RBF Kernel bandwidth
    )

    # 3. Run the online detection
    print("\nStarting online monitoring...")
    scores = []
    
    for t in range(total_length):
        score = model.update(data[t])
        scores.append(score)

    detected_cps = model.global_changepoints
    print(f"\nSimulation complete. Detected {len(detected_cps)} change points.")
    print(f"Final Regimes Stored: {len(model.regimes)}")

    # 4. Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Top Subplot: Input Signal ---
    ax1.plot(data, color="black", linewidth=0.8, alpha=0.7, label="Input Stream")
    
    # Plot true change points
    for i, cp in enumerate(actual_cps):
        label = "True Change" if i == 0 else ""
        ax1.axvline(x=cp, color="blue", linestyle="--", linewidth=2, label=label)
        
    # Plot detected change points (accounting for the algorithmic delay)
    for i, cp in enumerate(detected_cps):
        label = "Detected Change" if i == 0 else ""
        ax1.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)

    ax1.set_title("Open-World Stream (Mean & Variance Shifts)")
    ax1.set_ylabel("Signal Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Bottom Subplot: MMD Score ---
    scores = np.array(scores)
    
    ax2.plot(scores, color="teal", linewidth=1.5, label="Empirical MMD Score")
    
    # Plot the threshold line
    ax2.axhline(y=eta_threshold, color="orange", linestyle="--", linewidth=2, label=f"Detection Threshold ($\eta$ = {eta_threshold})")
    
    # Plot vertical lines for confirmed detections
    for cp in detected_cps:
        # The spike actually happened slightly before the confirmation time
        ax2.axvline(x=cp, color="red", linestyle="-", linewidth=2)

    ax2.set_title("Sliding Window MMD over Time")
    ax2.set_xlabel("Time Step (t)")
    ax2.set_ylabel("Squared MMD")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()