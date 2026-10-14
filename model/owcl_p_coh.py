import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class HybridOWRCPD:
    """
    Hybrid Open-World Change-Point Detection.
    - Detection: Pearson Divergence (Exact O(N^2) empirical operator norm)
    - Classification: MMD Coherence (Normalized RKHS overlap)
    - Kernel: Optional Dynamic Median Heuristic bandwidth tuning per regime.
    - Distance: Optional Hypersphere Normalization for scale invariance.
    """
    def __init__(
        self, 
        n: int, 
        m: int, 
        tau: float, 
        nu: float, 
        k_conf: int, 
        reg_lambda: float = 1e-5, 
        gamma: float = 0.1, 
        factor: float = 1.0,
        dynamic_gamma: bool = True,
        normalize_distance: bool = False
    ):
        # Window Parameters
        self.n = n                  
        self.m = m                  
        
        # Thresholds & Constraints
        self.tau = tau              # Detection threshold (Pearson)
        self.nu = nu                # Novelty threshold (MMD Coherence)
        self.k_conf = k_conf        
        
        # Kernel Parameters
        self.reg_lambda = reg_lambda 
        self.gamma = gamma          # Initial/Static gamma
        self.factor = factor        # Scaling factor for the median heuristic
        self.dynamic_gamma = dynamic_gamma
        self.normalize_distance = normalize_distance
        
        # System State
        self.t = 0
        self.state = "INIT_REGIME" 
        self.consec = 0
        
        # Data Buffers
        self.X: List[np.ndarray] = []                 
        self.Y: List[np.ndarray] = []                 
        self.X_prime: List[np.ndarray] = []           
        
        # Active Monitoring Tensors
        self.active_W = np.array([])
        self.active_h = np.array([])
        self.active_mu_K = np.array([])
        
        # Regimes Library: [{'X': array, 'gamma': float}]
        self.regimes: List[Dict[str, Any]] = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints: List[int] = []
        self.hybrid_scores: List[float] = []
        self.regime_log: List[int] = []

    def _sq_distances_v0(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Computes the squared Euclidean distance matrix."""
        X, Y = np.atleast_2d(X), np.atleast_2d(Y)
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
        dist_sq = X_sq + Y_sq - 2.0 * (X @ Y.T)
        
        return np.maximum(dist_sq, 0.0)

    def _sq_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Computes scale-invariant squared Euclidean distance by unit hypersphere projection."""
        X, Y = np.atleast_2d(X), np.atleast_2d(Y)
        
        # Internal Normalization (makes the model scale-invariant)
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        
        # Distance in Cosine space
        dist_sq = 2.0 - 2.0 * (X_norm @ Y_norm.T)
        
        return np.maximum(dist_sq, 0.0)

    def _get_active_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Routes to the appropriate distance metric based on the normalization toggle."""
        if self.normalize_distance:
            return self._sq_distances(X, Y)
        return self._sq_distances_v0(X, Y)

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma_override: Optional[float] = None) -> np.ndarray:
        """Gaussian RBF kernel matrix computation."""
        dist_sq = self._get_active_distances(X, Y)
        active_gamma = gamma_override if gamma_override is not None else self.gamma
        return np.exp(-active_gamma * dist_sq)

    def _compute_full_W_and_update_gamma(self, X_data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        1. Computes squared distances.
        2. Applies the Median Heuristic to tune gamma (if enabled).
        3. Computes exact inverse metric tensor W and reference mean vector mu_K.
        """
        X_arr = np.array(X_data)
        dist_sq = self._get_active_distances(X_arr, X_arr)
        
        if self.dynamic_gamma:
            # Extract only the unique pairwise distances (upper triangle)
            triu_indices = np.triu_indices_from(dist_sq, k=1)
            pairwise_sq_dists = dist_sq[triu_indices]
            
            median_sq_dist = np.median(pairwise_sq_dists)
            median_sq_dist = max(median_sq_dist, 1e-5) # Prevent division by zero
                
            self.gamma = self.factor / (2.0 * median_sq_dist)
            print(f"  [Debug] Gamma Heuristic: gamma={self.gamma:.6f} | Variance={np.var(pairwise_sq_dists):.4f}")
        else:
            print(f"  [Debug] Using Static Gamma: gamma={self.gamma:.6f}")
        
        # Compute K_XX using the active gamma
        K_XX = np.exp(-self.gamma * dist_sq)
        
        # Compute exact inverse metric tensor for Pearson tracking
        W = np.linalg.inv(K_XX + self.reg_lambda * np.eye(len(X_arr)))
        
        # 3. Mean of the reference kernel (mu_X in RKHS)
        mu_K = np.mean(K_XX, axis=1)
        
        return W, mu_K, self.gamma

    def _compute_rkhs_coherence(self, Z: List[np.ndarray], regime_dict: Dict[str, Any]) -> float:
        """
        Computes Normalized MMD Coherence for Classification.
        Crucially uses the historical regime's specific gamma.
        """
        X_i = regime_dict['X']
        gamma_i = regime_dict['gamma']
        
        K_AA = max(1e-12, np.mean(self._rbf_kernel(Z, Z, gamma_override=gamma_i)))
        K_BB = max(1e-12, np.mean(self._rbf_kernel(X_i, X_i, gamma_override=gamma_i)))
        K_AB = np.mean(self._rbf_kernel(Z, X_i, gamma_override=gamma_i))
        
        return float(K_AB / (np.sqrt(K_AA) * np.sqrt(K_BB)))

    def update(self, x: Any) -> float:
        self.t += 1
        x = np.atleast_1d(x)
        score = np.nan
        
        # --- PHASE 1: Initialize First Regime ---
        if self.state == "INIT_REGIME":
            self.X.append(x)
            self.regime_log.append(-1)
            
            if len(self.X) == self.n:
                W_init, mu_K_init, gamma_init = self._compute_full_W_and_update_gamma(self.X)
                self.regimes.append({'X': list(self.X), 'gamma': gamma_init})
                
                self.active_W = W_init.copy()
                self.active_mu_K = mu_K_init.copy()
                self.active_regime_id = 0
                self.state = "WARMUP_Y"
                
        # --- PHASE 2: Fill Analysis Window ---
        elif self.state == "WARMUP_Y":
            self.Y.append(x)
            self.regime_log.append(self.active_regime_id)
            
            if len(self.Y) == self.m:
                self.active_h = np.mean(self._rbf_kernel(self.X, self.Y), axis=1)
                self.state = "ACTIVE"
                
        # --- PHASE 3: Pearson Active Monitoring (O(N^2) Sliding Window) ---
        elif self.state == "ACTIVE":
            x_t, x_t_m = x, self.Y[0] 
            
            # 1. Downdate W
            w11, w12 = self.active_W[0, 0], self.active_W[0, 1:]
            w21, W22 = self.active_W[1:, 0], self.active_W[1:, 1:]
            W_reduced = W22 - np.outer(w21, w12) / w11
            
            # 2. Update W
            X_reduced = self.X[1:]  
            k_in = self._rbf_kernel(X_reduced, x_t_m).flatten()
            
            v = W_reduced @ k_in
            schur_scalar = max(1e-12, (1.0 + self.reg_lambda) - (k_in @ v))
            beta = 1.0 / schur_scalar
            
            self.active_W = np.block([
                [W_reduced + beta * np.outer(v, v), -beta * v[:, None]],
                [-beta * v[None, :],                np.array([[beta]])]
            ])

            # 3. Update h vector recursively (Fixed Order)
            h_reduced = self.active_h[1:].copy() 
            k_in_xt = self._rbf_kernel(X_reduced, x_t).flatten()
            h_reduced += (1.0 / self.m) * (k_in_xt - k_in)
            
            # Calculate h_new BEFORE sliding the buffers
            h_new = np.mean(self._rbf_kernel(x_t_m, self.Y[1:] + [x_t]).flatten())
            self.active_h = np.append(h_reduced, h_new)
            
            # Slide physical buffers
            self.Y.pop(0)
            self.Y.append(x_t)
            self.X.pop(0)
            self.X.append(x_t_m)
            self.regime_log.append(self.active_regime_id)
            
            # 4. Evaluate Pearson Divergence Score (Exact NeurIPS Evaluation)
            self.active_mu_K = np.mean(self._rbf_kernel(self.X, self.X), axis=1)
            delta = self.active_h - self.active_mu_K
            score = 0.5 * float((self.active_W @ delta) @ delta)
            
            self.consec = self.consec + 1 if score > self.tau else 0
                
            if self.consec >= self.k_conf:
                self.global_changepoints.append(self.t)
                print(f"\nTime {self.t}: *** CHANGE CONFIRMED *** (Hybrid Pearson Div={score:.4f} > {self.tau})")
                
                self.consec = 0
                self.X_prime.clear()
                self.state = "COLLECT_PURE_WINDOW"
                
        # --- PHASE 4: MMD OWR Classification ---
        elif self.state == "COLLECT_PURE_WINDOW":
            self.X_prime.append(x)
            self.regime_log.append(-1)
            
            if len(self.X_prime) == self.n:
                print(f"\n  [Classification Phase @ Time {self.t}]")
                
                # Compare against regimes using MMD Coherence
                coherence_scores = [self._compute_rkhs_coherence(self.X_prime, r) for r in self.regimes]
                for i, coh in enumerate(coherence_scores):
                    print(f"    - vs Regime {i}: MMD Coherence = {coh:.4f}")
                
                best_i = np.argmax(coherence_scores) if coherence_scores else -1
                max_coh = coherence_scores[best_i] if coherence_scores else -float('inf')
                        
                # Compute new W and get the newly tuned gamma for the fresh window
                new_W, new_mu_K, new_gamma = self._compute_full_W_and_update_gamma(self.X_prime) 
                
                # Classification Decision (Note: MMD Coherence uses '>=', not '<=')
                if max_coh >= self.nu:
                    print(f"  -> RESULT: Assigned to KNOWN Regime {best_i} (Max = {max_coh:.4f} >= {self.nu})\n")
                    self.active_regime_id = best_i
                    self.regimes[best_i] = {'X': list(self.X_prime), 'gamma': new_gamma}
                else:
                    self.active_regime_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {self.active_regime_id} (Max = {max_coh:.4f} < {self.nu})\n")
                    self.regimes.append({'X': list(self.X_prime), 'gamma': new_gamma})
                
                # Restart the Pearson sliding window
                self.X = list(self.X_prime)
                self.active_W = new_W.copy()
                self.active_mu_K = new_mu_K.copy()
                self.Y.clear()
                self.state = "WARMUP_Y"
                
        self.hybrid_scores.append(score)
        return score