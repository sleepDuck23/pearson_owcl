import numpy as np

class OWR_CPD:
    def __init__(self, n, m, eta, nu, k_conf, gamma=0.1, factor=1.0):
        """
        Open-World Change-Point Detection with Recurrent Regime Recall.
        Based on MMD detection and RKHS coherence for regime assignment.
        - Kernel: Dynamic Median Heuristic bandwidth tuning per regime.
        """
        self.n = n              # Reference window size
        self.m = m              # Analysis window size
        self.eta = eta          # Detection threshold (MMD)
        self.nu = nu            # Novelty threshold (Coherence)
        self.k_conf = k_conf    # Consecutive windows required to confirm change
        self.gamma = gamma      # Will be dynamically overwritten by the median heuristic
        self.factor = factor    # Scaling factor for gamma adjustment (if needed)
        
        # System State
        self.t = 0
        self.state = "INIT_REGIME" 
        self.consec = 0
        
        # Buffers
        self.X = []             # Reference window (length n)
        self.Y = []             # Analysis window (length m)
        self.X_prime = []       # Pure representative window after change (length n)
        
        # Regimes Dictionary: List of dicts {'X': array, 'gamma': float}
        self.regimes = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints = []
        self.mmd_scores = []
        self.regime_log = []

    def _sq_distances(self, X, Y):
        """Computes the squared Euclidean distance matrix."""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        X2 = np.sum(X**2, axis=1).reshape(-1, 1)
        Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
        dist_sq = X2 + Y2 - 2 * np.dot(X, Y.T)
        return np.maximum(dist_sq, 0.0)

    def _rbf_kernel(self, X, Y, gamma_override=None):
        """Gaussian RBF kernel matrix computation."""
        dist_sq = self._sq_distances(X, Y)
        g = gamma_override if gamma_override is not None else self.gamma
        return np.exp(-g * dist_sq)

    def _update_gamma_median_heuristic(self, X_data):
        """
        Computes the squared distances and applies the Median Heuristic
        to dynamically tune the active gamma for the current regime.
        """
        X_arr = np.array(X_data)
        dist_sq = self._sq_distances(X_arr, X_arr)
        
        # Extract only the unique pairwise distances (upper triangle, ignoring diagonal)
        triu_indices = np.triu_indices_from(dist_sq, k=1)
        pairwise_sq_dists = dist_sq[triu_indices]
        
        # Apply Median Heuristic
        median_sq_dist = np.median(pairwise_sq_dists)
        if median_sq_dist == 0:
            median_sq_dist = 1e-5 # Prevent division by zero
            
        self.gamma = self.factor * (1.0 / (2.0 * median_sq_dist))
        return self.gamma

    def _compute_mmd_squared(self, X, Y):
        """Computes the empirical squared MMD between two windows using the active gamma."""
        K_XX = np.mean(self._rbf_kernel(X, X))
        K_YY = np.mean(self._rbf_kernel(Y, Y))
        K_XY = np.mean(self._rbf_kernel(X, Y))
        return max(0.0, K_XX + K_YY - 2 * K_XY)

    def _compute_coherence(self, X_prime, regime_dict):
        """
        Computes the normalized RKHS alignment between two mean embeddings.
        Crucially uses the historical regime's specific gamma.
        """
        regime_data = regime_dict['X']
        gamma_i = regime_dict['gamma']
        
        K_AA = np.mean(self._rbf_kernel(X_prime, X_prime, gamma_override=gamma_i))
        K_BB = np.mean(self._rbf_kernel(regime_data, regime_data, gamma_override=gamma_i))
        K_AB = np.mean(self._rbf_kernel(X_prime, regime_data, gamma_override=gamma_i))
        
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
                # Dynamically tune gamma for the first regime
                gamma_init = self._update_gamma_median_heuristic(self.X)
                
                # Store initial regime representative as a dictionary
                self.regimes.append({'X': list(self.X), 'gamma': gamma_init})
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
                for i, regime_dict in enumerate(self.regimes):
                    coh = self._compute_coherence(self.X_prime, regime_dict)
                    coherence_scores.append(coh)
                    print(f"    - vs Regime {i}: MMD Coherence = {coh:.4f}")
                
                # Identify the best match
                if coherence_scores:
                    best_i = np.argmax(coherence_scores)
                    max_coh = coherence_scores[best_i]
                else:
                    best_i = -1
                    max_coh = -float('inf')
                        
                # Profile the new window & extract the newly tuned gamma
                new_gamma = self._update_gamma_median_heuristic(self.X_prime)
                
                # Classification Decision
                if max_coh >= self.nu:
                    print(f"  -> RESULT: Assigned to KNOWN Regime {best_i} (Max = {max_coh:.4f} >= {self.nu})\n")
                    self.active_regime_id = best_i
                    self.regimes[best_i] = {'X': list(self.X_prime), 'gamma': new_gamma}
                else:
                    new_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {new_id} (Max = {max_coh:.4f} < {self.nu})\n")
                    self.active_regime_id = new_id
                    self.regimes.append({'X': list(self.X_prime), 'gamma': new_gamma})
                
                # Restart Active Monitoring
                self.X = list(self.X_prime)
                self.Y = []
                self.state = "WARMUP_Y"
                
        self.mmd_scores.append(score)
        return score