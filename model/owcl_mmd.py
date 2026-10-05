import numpy as np

class OWR_CPD:
    def __init__(self, n, m, eta, nu, k_conf, gamma=0.1):
        """
        Open-World Change-Point Detection with Recurrent Regime Recall.
        Based on MMD detection and RKHS coherence for regime assignment.
        """
        self.n = n              # Reference window size
        self.m = m              # Analysis window size
        self.eta = eta          # Detection threshold (MMD)
        self.nu = nu            # Novelty threshold (Coherence)
        self.k_conf = k_conf    # Consecutive windows required to confirm change
        self.gamma = gamma      # RBF kernel bandwidth
        
        # System State
        self.t = 0
        self.state = "INIT_REGIME" 
        self.consec = 0
        
        # Buffers
        self.X = []             # Reference window (length n)
        self.Y = []             # Analysis window (length m)
        self.X_prime = []       # Pure representative window after change (length n)
        
        # Regimes Dictionary (storing the raw windows to represent mean embeddings)
        self.regimes = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints = []
        self.mmd_scores = []
        self.regime_log = []

    def _rbf_kernel(self, X, Y):
        """Gaussian RBF kernel matrix computation."""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        X2 = np.sum(X**2, axis=1).reshape(-1, 1)
        Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
        return np.exp(-self.gamma * (X2 + Y2 - 2 * np.dot(X, Y.T)))

    def _compute_mmd_squared(self, X, Y):
        """Computes the empirical squared MMD between two windows."""
        K_XX = np.mean(self._rbf_kernel(X, X))
        K_YY = np.mean(self._rbf_kernel(Y, Y))
        K_XY = np.mean(self._rbf_kernel(X, Y))
        return max(0.0, K_XX + K_YY - 2 * K_XY)

    def _compute_coherence(self, X_prime, regime_data):
        """Computes the normalized RKHS alignment between two mean embeddings."""
        K_AA = np.mean(self._rbf_kernel(X_prime, X_prime))
        K_BB = np.mean(self._rbf_kernel(regime_data, regime_data))
        K_AB = np.mean(self._rbf_kernel(X_prime, regime_data))
        
        norm_A = np.sqrt(max(1e-12, K_AA))
        norm_B = np.sqrt(max(1e-12, K_BB))
        return K_AB / (norm_A * norm_B)

    def update(self, x):
        """Main online streaming process."""
        self.t += 1
        x = np.atleast_1d(x)
        score = np.nan
        
        # --- PHASE 1: Initialize First Regime ---
        if self.state == "INIT_REGIME":
            self.X.append(x)
            self.regime_log.append(-1)
            
            if len(self.X) == self.n:
                # Store initial regime representative (as a pure list)
                self.regimes.append(list(self.X))
                self.active_regime_id = 0
                self.state = "WARMUP_Y"
                
        # --- PHASE 2: Fill Analysis Window ---
        elif self.state == "WARMUP_Y":
            self.Y.append(x)
            self.regime_log.append(self.active_regime_id)
            
            if len(self.Y) == self.m:
                self.state = "ACTIVE"
                
        # --- PHASE 3: Active Monitoring ---
        elif self.state == "ACTIVE":
            # Slide windows forward
            x_out = self.Y.pop(0)
            self.X.append(x_out)
            self.X.pop(0)
            self.Y.append(x)
            
            self.regime_log.append(self.active_regime_id)
            
            # Evaluate MMD (Eq. 4)
            score = self._compute_mmd_squared(self.X, self.Y)
            
            if score > self.eta:
                self.consec += 1
            else:
                self.consec = 0
                
            if self.consec >= self.k_conf:
                # Confirmed Change Point
                cp_time = self.t
                self.global_changepoints.append(cp_time)
                print(f"\nTime {cp_time} (detected at {self.t}): *** CHANGE CONFIRMED *** (MMD={score:.4f} > {self.eta})")
                
                self.consec = 0
                self.X_prime = []
                self.state = "COLLECT_PURE_WINDOW"
                
        # --- PHASE 4: OWR Classification ---
        elif self.state == "COLLECT_PURE_WINDOW":
            self.X_prime.append(x)
            self.regime_log.append(-1)
            
            if len(self.X_prime) == self.n:
                print(f"\n  [Classification Phase @ Time {self.t}]")
                coherence_scores = []
                
                # Check coherence against all stored regimes
                for i, regime_data in enumerate(self.regimes):
                    coh = self._compute_coherence(self.X_prime, regime_data)
                    coherence_scores.append(coh)
                    print(f"    - vs Regime {i}: MMD Coherence = {coh:.4f}")
                
                # Identify the best match
                if coherence_scores:
                    best_i = np.argmax(coherence_scores)
                    max_coh = coherence_scores[best_i]
                else:
                    best_i = -1
                    max_coh = -float('inf')
                        
                # Classification Decision
                if max_coh >= self.nu:
                    print(f"  -> RESULT: Assigned to KNOWN Regime {best_i} (Max = {max_coh:.4f} >= {self.nu})\n")
                    self.active_regime_id = best_i
                    self.regimes[best_i] = list(self.X_prime)
                else:
                    new_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {new_id} (Max = {max_coh:.4f} < {self.nu})\n")
                    self.active_regime_id = new_id
                    self.regimes.append(list(self.X_prime))
                
                # Restart Active Monitoring
                self.X = list(self.X_prime)
                self.Y = []
                self.state = "WARMUP_Y"
                
        self.mmd_scores.append(score)
        return score