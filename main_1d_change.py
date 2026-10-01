import numpy as np
import matplotlib.pyplot as plt

# Import the data generator and the model
from data.generation import generate_mean_shift_data
from model.cpd_x2 import Pearson_CPD

def main():
    # 1. Generate the stream data
    total_length = 3000
    true_cps = [1000, 2000]  # True change points for the simulation
    
    print("Generating data...")
    data, actual_cps = generate_mean_shift_data(
        length=total_length, 
        changepoints=true_cps, 
        means=[0.0, 3.0, -1.0], # Explicit means to ensure clear shifts
        std=1.0, 
        seed=42
    )

    # 2. Initialize the Pearson CPD model
    # Window sizes (n=100, m=50) and tau depend on your specific data variance
    tau_threshold = 0.2  # Detection threshold
    model = Pearson_CPD(
        n=100, 
        m=50, 
        tau=0.2, 
        reg_lambda=1e-5, 
        gamma=0.05
    )

    # 3. Run the online detection
    print("\nStarting online monitoring...")
    scores = []
    
    for t in range(total_length):
        # Feed one sample at a time
        score = model.update(data[t])
        scores.append(score)

    detected_cps = model.global_changepoints
    print(f"\nSimulation complete. Detected {len(detected_cps)} change points.")

    # 4. Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Top Subplot: Input Signal ---
    ax1.plot(data, color="black", linewidth=0.8, alpha=0.7, label="Input Stream")
    
    # Plot true change points
    for i, cp in enumerate(actual_cps):
        label = "True Change" if i == 0 else ""
        ax1.axvline(x=cp, color="blue", linestyle="--", linewidth=2, label=label)
        
    # Plot detected change points
    for i, cp in enumerate(detected_cps):
        label = "Detected Change" if i == 0 else ""
        ax1.axvline(x=cp, color="red", linestyle="-", linewidth=2, label=label)

    ax1.set_title("Time-Series Stream with Change Points")
    ax1.set_ylabel("Signal Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Bottom Subplot: Pearson Score ---
    # Convert scores to numpy array for plotting (handles NaNs from warm-up automatically)
    scores = np.array(scores)
    
    ax2.plot(scores, color="purple", linewidth=1.5, label="Pearson Divergence Score")
    
    # Plot the threshold line
    ax2.axhline(y=tau_threshold, color="green", linestyle="--", linewidth=2, label=f"Threshold (tau = {tau_threshold})")
    
    # Plot vertical lines where the threshold was crossed (Detection triggered)
    for i, cp in enumerate(detected_cps):
        ax2.axvline(x=cp, color="red", linestyle="-", linewidth=2)

    ax2.set_title("Pearson CPD Score over Time")
    ax2.set_xlabel("Time Step (t)")
    ax2.set_ylabel("Divergence Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()