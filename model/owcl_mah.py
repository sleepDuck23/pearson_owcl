import numpy as np
from .owcl_pearson import (validate_parameters, squared_distances,
                    inverse_gram, slide_inverse, operator_distance, pearson_score)
from typing import List, Dict, Tuple, Optional, Any

class MahalanobisOWRCPD:
    """
    Open-World Change-Point Detection with PROPER Kernel Mahalanobis Classification.
    
    - Detection: Amortized O(n^2 + m^2) recursive Pearson Divergence (regularized RKHS second-moment operator).
    - Classification: Squared Mahalanobis distance using centered T_i.

    Detection uses uncentered T_i = mean(phi(X) tensor phi(X)).
    Classification uses centered T_i = mean((phi(X)-mu_i) tensor (phi(X)-mu_i)),
    regularized by reg_lambda * I. Both operators use normalization 1/n.
    - Kernel: Dynamic Median Absolute Deviation (MAD) bandwidth tuning per regime.
    - Distance: Optional Hypersphere Normalization for scale invariance.
    """
    
    def __init__(
        self, 
        n: int, 
        m: int, 
        tau: float, 
        nu: float, 
        k_conf: int, 
        reg_lambda: float = 1e-3, 
        gamma: float = 0.1, 
        factor: float = 1.0,
        dynamic_gamma: bool = True,
        normalize_distance: bool = False
    ):
        # Window Parameters
        validate_parameters(n, m, k_conf, gamma, factor, reg_lambda)
        self.n = n                  # Reference window size
        self.m = m                  # Analysis window size
        
        # Thresholds & Constraints
        self.tau = tau              # Detection threshold (Pearson Divergence)
        self.nu = nu                # Novelty threshold (Max Squared Kernel Mahalanobis Distance)
        self.k_conf = k_conf        # Consecutive windows to confirm change
        
        # Kernel Parameters
        self.reg_lambda = reg_lambda 
        self.gamma = gamma          # Initial/Static gamma
        self.factor = factor        # Scaling factor for the MAD heuristic
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
        
        # Regimes Library: Stores Centered & Uncentered parameters and active gamma
        self.regimes: List[Dict[str, Any]] = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints: List[int] = []
        self.pearson_div_scores: List[float] = []
        self.regime_log: List[int] = []

    def _sq_distances_v0(self, X, Y):
        return squared_distances(X, Y, normalize=False)

    def _sq_distances(self, X, Y):
        return squared_distances(X, Y, normalize=True)

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

    def _compute_regime_tensors(self, X_data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        1. Applies the MAD Guardrail Heuristic to tune gamma robustly (if enabled).
        2. Computes uncentered tensors (for recursive active tracking).
        3. Computes centered tensors (for proper Mahalanobis Classification).
        """
        X_arr = np.array(X_data)
        
        if self.dynamic_gamma:
            # Flatten to 1D for MAD computation
            X_flat = X_arr.flatten()
            
            # 1. Compute spread using MAD (robust to zero-centering and outliers)
            median_val = np.median(X_flat)
            mad = np.median(np.abs(X_flat - median_val))
            
            # 2. Prevent division by zero if window is completely flat
            if mad < 1e-4:
                mad = 1.0
                
            # 3. Convert MAD to standard deviation scale (1.4826 normal consistency factor)
            sigma_scale = mad * 1.4826 
            
            # 4. Set Gamma (Bandwidth) based on physical spread
            raw_gamma = self.factor / (2.0 * (sigma_scale ** 2))
            
            # 5. Safety Clip to prevent matrix singularity or explosion
            self.gamma = np.clip(raw_gamma, 1e-3, 50.0)
            
        # Calculate Squared Distances (for the kernel matrix)
        dist_sq = self._get_active_distances(X_arr, X_arr)
        
        # Compute K_XX using the active gamma
        K_XX = np.exp(-self.gamma * dist_sq)
        
        # 1. Uncentered W (Used for continuous sliding window Pearson Divergence)
        W_uncentered = inverse_gram(K_XX, self.reg_lambda)
        
        # 2. Centered W_c (Used for proper Kernel Mahalanobis Distance)
        # H K_XX H, where H = I - ones((n,n))/n.
        K_c = (K_XX - K_XX.mean(axis=0)[None, :]
               - K_XX.mean(axis=1)[:, None] + K_XX.mean())
        W_centered = inverse_gram(K_c, self.reg_lambda)
        
        # 3. Mean of the reference kernel (mu_X in RKHS)
        mu_K = np.mean(K_XX, axis=1)
        
        return W_uncentered, W_centered, mu_K, self.gamma

    def _compute_kernel_mahalanobis_distance(self, Z: List[np.ndarray], regime_dict: Dict[str, Any]) -> float:
        """
        Compute <mu_Z-mu_X, (T_i_centered + lambda I)^-1 (mu_Z-mu_X)>.
        This squared Mahalanobis distance has no Pearson factor of 1/2.
        """
        X_i, W_c = regime_dict['X'], regime_dict['W_c']
        mu_K, gamma_i = regime_dict['mu_K'], regime_dict['gamma']
        
        # 1. Compute uncentered cross-kernel mean vector using historical gamma
        h = np.mean(self._rbf_kernel(X_i, Z, gamma_override=gamma_i), axis=1)
        
        # 2. Exact RKHS Matrix Centering (Projects the difference into the centered space)
        delta = h - mu_K
        h_tilde = delta - np.mean(delta)
        
        # 3. Compute True Mahalanobis Distance (Mahalanobis norm in centered RKHS)
        mmd_squared = (np.mean(mu_K)
                       + np.mean(self._rbf_kernel(Z, Z, gamma_override=gamma_i))
                       - 2 * np.mean(h))
        dist_sq = operator_distance(mmd_squared, h_tilde, W_c, self.reg_lambda)
        
        return dist_sq

    def update(self, x: Any) -> float:
        """Main online streaming process."""
        self.t += 1
        x = np.atleast_1d(x)
        score = np.nan
        
        # --- PHASE 1: Initialize First Regime ---
        if self.state == "INIT_REGIME":
            self.X.append(x)
            self.regime_log.append(-1)
            
            if len(self.X) == self.n:
                W_u, W_c, mu_K, gamma_init = self._compute_regime_tensors(self.X)
                
                self.regimes.append({
                    'X': list(self.X), 
                    'W_uncentered': W_u, 
                    'W_c': W_c, 
                    'mu_K': mu_K,
                    'gamma': gamma_init
                })
                
                self.active_W = W_u.copy() 
                self.active_mu_K = mu_K.copy()
                self.active_regime_id = 0
                self.state = "WARMUP_Y"
                
        # --- PHASE 2: Fill Analysis Window ---
        elif self.state == "WARMUP_Y":
            self.Y.append(x)
            self.regime_log.append(self.active_regime_id)
            
            if len(self.Y) == self.m:
                self.active_h = np.mean(self._rbf_kernel(self.X, self.Y), axis=1)
                self.state = "ACTIVE"
                
        # --- PHASE 3: Active Monitoring (Uncentered Pearson) ---
        elif self.state == "ACTIVE":
            x_t, x_t_m = x, self.Y[0]
            
            # W = (K_XX + n * lambda I)^-1, with fixed gamma.
            self.active_W = slide_inverse(self, x_t_m)
            X_reduced = self.X[1:]
            k_in = self._rbf_kernel(X_reduced, x_t_m).flatten()

            # 3. Update h vector
            h_reduced = self.active_h[1:].copy() 
            k_in_xt = self._rbf_kernel(X_reduced, x_t).flatten()
            h_reduced += (1.0 / self.m) * (k_in_xt - k_in)
            
            h_new = np.mean(self._rbf_kernel(x_t_m, self.Y[1:] + [x_t]).flatten())
            self.active_h = np.append(h_reduced, h_new)
            
            # Slide physical buffers
            self.Y.pop(0)
            self.Y.append(x_t)
            self.X.pop(0)
            self.X.append(x_t_m)
            self.regime_log.append(self.active_regime_id)
            
            # 4. Evaluate Divergence Score (regularized RKHS Pearson-type discrepancy)
            # Recompute active_mu_K for the shifted reference window 
            # Reference Gram evaluation costs O(n^2) kernel evaluations.
            self.active_mu_K = np.mean(self._rbf_kernel(self.X, self.X), axis=1)
            
            score = pearson_score(
                self, self.Y, self.X, self.active_W, self.active_mu_K,
                self.gamma, h=self.active_h)
            
            self.consec = self.consec + 1 if score > self.tau else 0
                
            if self.consec >= self.k_conf:
                self.global_changepoints.append(self.t)
                print(f"\nTime {self.t}: *** CHANGE CONFIRMED *** (Pearson Div={score:.4f} > {self.tau})")
                
                self.consec = 0
                self.X_prime.clear()
                self.state = "COLLECT_PURE_WINDOW"
                
        # --- PHASE 4: OWR Classification (Proper Kernel Mahalanobis) ---
        elif self.state == "COLLECT_PURE_WINDOW":
            self.X_prime.append(x)
            self.regime_log.append(-1)
            
            # Ensure the test window Z is exactly the size of the reference window X
            if len(self.X_prime) == self.n:
                print(f"\n  [Classification Phase @ Time {self.t}]")
                
                # Compare new distribution Z against all historical distributions X_i
                mah_distances = [self._compute_kernel_mahalanobis_distance(self.X_prime, r) for r in self.regimes]
                for i, dist_sq in enumerate(mah_distances):
                    print(f"    - vs Regime {i}: KMD^2 = {dist_sq:.4f}")
                
                best_i = np.argmin(mah_distances) if mah_distances else -1
                min_dist = mah_distances[best_i] if mah_distances else float('inf')
                        
                # Pre-calculate Tensors and active gamma for the new pure window
                W_u, W_c, mu_K, new_gamma = self._compute_regime_tensors(self.X_prime)
                new_regime_dict = {
                    'X': list(self.X_prime), 
                    'W_uncentered': W_u, 
                    'W_c': W_c, 
                    'mu_K': mu_K,
                    'gamma': new_gamma
                }
                
                # Classification Decision based on Novelty Threshold (nu)
                if min_dist <= self.nu:
                    print(f"  -> RESULT: Assigned to KNOWN Regime {best_i} (Min KMD^2 = {min_dist:.4f} <= {self.nu})\n")
                    self.active_regime_id = best_i
                    self.regimes[best_i] = new_regime_dict
                else:
                    self.active_regime_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {self.active_regime_id} (Min KMD^2 = {min_dist:.4f} > {self.nu})\n")
                    self.regimes.append(new_regime_dict)
                
                # Restart Active Monitoring
                self.X = list(self.X_prime)
                self.active_W = W_u.copy()
                self.active_mu_K = mu_K.copy()
                self.Y.clear()
                self.state = "WARMUP_Y"
                
        self.pearson_div_scores.append(score)
        return score
