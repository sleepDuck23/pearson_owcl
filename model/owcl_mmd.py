import numpy as np

def validate_parameters(n, m, k_conf, gamma, factor, reg_lambda=None):
    for name, value in (("n", n), ("m", m), ("k_conf", k_conf)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("gamma", gamma), ("factor", factor),
                        ("reg_lambda", reg_lambda)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")


def squared_distances(X, Y, normalize=False):
    """Rows are observations; a 1D array represents one observation."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    nx = 0 if X.size == 0 else np.atleast_2d(X).shape[0]
    ny = 0 if Y.size == 0 else np.atleast_2d(Y).shape[0]
    if not nx or not ny:
        return np.empty((nx, ny))
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    if normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    # Actual norms also handle zero vectors, whose normalized norm is not one.
    return np.maximum(np.sum(X * X, axis=1)[:, None]
                      + np.sum(Y * Y, axis=1)[None, :] - 2 * np.dot(X, Y.T), 0.0)


def median_gamma(distances, factor, fallback):
    pairs = distances[np.triu_indices_from(distances, k=1)]
    if not pairs.size:
        return float(fallback)
    return float(factor / (2 * max(float(np.median(pairs)), 1e-5)))


class OWR_CPD:
    def __init__(self, n, m, eta, nu, k_conf, gamma=0.1, factor=1.0, dynamic_gamma=True):
        """
        Open-World Change-Point Detection with Recurrent Regime Recall.
        Biased empirical MMD squared detection and RKHS coherence classification.
        eta thresholds MMD squared; nu thresholds cosine similarity of the
        mean embeddings (higher means closer).
        - Kernel: Optional median-heuristic bandwidth tuning per regime.
        """
        validate_parameters(n, m, k_conf, gamma, factor)
        self.dynamic_gamma = dynamic_gamma
        self.n = n              # Reference window size
        self.m = m              # Analysis window size
        self.eta = eta          # Detection threshold (MMD squared)
        self.nu = nu            # Novelty threshold (Coherence)
        self.k_conf = k_conf    # Consecutive windows required to confirm change
        self.gamma = gamma      # Updated by the heuristic only if dynamic_gamma=True
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
        return squared_distances(X, Y, normalize=False)

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
        if self.dynamic_gamma:
            distances = self._sq_distances(X_data, X_data)
            self.gamma = median_gamma(distances, self.factor, self.gamma)
        return self.gamma

    def _compute_mmd_squared(self, X, Y):
        """Computes the biased empirical squared MMD (V-statistic) between two windows using the active gamma."""
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
        return float(np.clip(K_AB / (norm_A * norm_B), 0.0, 1.0))

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
