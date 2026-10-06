import numpy as np

class Mahalanobis_OWR_CPD:
    def __init__(self, n, m, tau, nu, k_conf, reg_lambda=1e-3, gamma=0.1):
        """
        Open-World Change-Point Detection with PROPER Kernel Mahalanobis Classification.
        - Detection: O(N^2) recursive Pearson Divergence (Uncentered Kernel Space).
        - Classification: True Kernel Mahalanobis Distance (Centered RKHS).
        - Kernel: Dynamic Median Heuristic bandwidth tuning per regime.
        """
        self.n = n                  # Reference window size
        self.m = m                  # Analysis window size
        self.tau = tau              # Detection threshold (Pearson Divergence)
        self.nu = nu                # Novelty threshold (Max Squared Kernel Mahalanobis Distance)
        self.k_conf = k_conf        # Consecutive windows to confirm change
        self.reg_lambda = reg_lambda 
        self.gamma = gamma          # Will be dynamically overwritten by the median heuristic
        
        # System State
        self.t = 0
        self.state = "INIT_REGIME" 
        self.consec = 0
        
        # Buffers
        self.X = []                 
        self.Y = []                 
        self.X_prime = []           
        
        # Active Monitoring Tensors (Updated in O(N^2))
        self.active_W = np.array([])
        self.active_h = np.array([])
        
        # Regimes Library: Stores Centered & Uncentered parameters and active gamma
        self.regimes = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints = []
        self.pearson_div_scores = []
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

    def _compute_regime_tensors(self, X_data):
        """
        1. Computes squared distances and applies the Median Heuristic to tune gamma.
        2. Computes both uncentered tensors (for O(N^2) active tracking) 
           and centered tensors (for proper Mahalanobis Classification).
        """
        X_arr = np.array(X_data)
        n_samples = len(X_arr)
        
        # Calculate Squared Distances & apply Median Heuristic
        dist_sq = self._sq_distances(X_arr, X_arr)
        triu_indices = np.triu_indices_from(dist_sq, k=1)
        pairwise_sq_dists = dist_sq[triu_indices]
        
        median_sq_dist = np.median(pairwise_sq_dists)
        if median_sq_dist == 0:
            median_sq_dist = 1e-5 # Prevent division by zero
            
        self.gamma = 1.0 / (2.0 * median_sq_dist)
        
        # Compute K_XX using the newly tuned gamma
        K_XX = np.exp(-self.gamma * dist_sq)
        
        # 1. Uncentered W (Used for continuous sliding window Pearson Divergence)
        W_uncentered = np.linalg.inv(K_XX + self.reg_lambda * np.eye(n_samples))
        
        # 2. Centered W_c (Used for proper Kernel Mahalanobis Distance)
        # H is the centering matrix: H = I - (1/n) * 11^T
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples 
        K_c = H @ K_XX @ H
        W_centered = np.linalg.inv(K_c + self.reg_lambda * np.eye(n_samples))
        
        # 3. Mean of the reference kernel (mu_X in RKHS)
        mu_K = np.mean(K_XX, axis=1)
        
        return W_uncentered, W_centered, mu_K, self.gamma

    def _compute_kernel_mahalanobis_distance(self, Z, regime_dict):
        """
        Computes the True Kernel Mahalanobis Distance (squared) between 
        the mean embedding of the new window Z and the historical regime X.
        Crucially uses the historical regime's specific gamma.
        """
        X_i = regime_dict['X']
        W_c = regime_dict['W_c']
        mu_K = regime_dict['mu_K']
        gamma_i = regime_dict['gamma']
        
        # 1. Compute uncentered cross-kernel mean vector between X_i and Z using historical gamma
        h = np.mean(self._rbf_kernel(X_i, Z, gamma_override=gamma_i), axis=1)
        
        # 2. Center the distance vector perfectly in the RKHS.
        h_tilde_raw = h - mu_K
        h_tilde = h_tilde_raw - np.mean(h_tilde_raw) 
        
        # 3. Compute True Mahalanobis Distance (Mahalanobis norm in centered RKHS)
        dist_sq = np.dot(W_c @ h_tilde, h_tilde)
        
        return float(dist_sq)

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
                W_u, W_c, mu_K, gamma_init = self._compute_regime_tensors(self.X)
                
                self.regimes.append({
                    'X': list(self.X), 
                    'W_uncentered': W_u, 
                    'W_c': W_c, 
                    'mu_K': mu_K,
                    'gamma': gamma_init
                })
                
                self.active_W = W_u.copy() # Active monitoring requires uncentered W
                self.active_regime_id = 0
                self.state = "WARMUP_Y"
                
        # --- PHASE 2: Fill Analysis Window ---
        elif self.state == "WARMUP_Y":
            self.Y.append(x)
            self.regime_log.append(self.active_regime_id)
            
            if len(self.Y) == self.m:
                self.active_h = np.mean(self._rbf_kernel(self.X, self.Y), axis=1)
                self.state = "ACTIVE"
                
        # --- PHASE 3: Active Monitoring (Uncentered Pearson - O(N^2)) ---
        elif self.state == "ACTIVE":
            x_t = x                
            x_t_m = self.Y[0]      
            
            # 1. Downdate W
            w11 = self.active_W[0, 0]
            w12 = self.active_W[0, 1:]
            w21 = self.active_W[1:, 0]
            W22 = self.active_W[1:, 1:]
            W_reduced = W22 - np.outer(w21, w12) / w11
            
            # 2. Update W 
            X_reduced = self.X[1:]  
            k_in = self._rbf_kernel(X_reduced, x_t_m).flatten()
            v = W_reduced @ k_in
            schur_scalar = max(1e-12, (1.0 + self.reg_lambda) - np.dot(k_in, v))
            beta = 1.0 / schur_scalar
            
            self.active_W = np.block([
                [W_reduced + beta * np.outer(v, v), -beta * v[:, None]],
                [-beta * v[None, :],                np.array([[beta]])]
            ])
            
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
            
            # 4. Evaluate Divergence Score (Pearson)
            overlap = np.dot(self.active_W @ self.active_h, self.active_h)
            score = abs(overlap - 1.0) 
            
            if score > self.tau:
                self.consec += 1
            else:
                self.consec = 0
                
            if self.consec >= self.k_conf:
                cp_time = self.t
                self.global_changepoints.append(cp_time)
                print(f"\nTime {cp_time} (detected at {self.t}): *** CHANGE CONFIRMED *** (Pearson Div={score:.4f} > {self.tau})")
                
                self.consec = 0
                self.X_prime = []
                self.state = "COLLECT_PURE_WINDOW"
                
        # --- PHASE 4: OWR Classification (Proper Kernel Mahalanobis) ---
        elif self.state == "COLLECT_PURE_WINDOW":
            self.X_prime.append(x)
            self.regime_log.append(-1)
            
            # Ensure the test window Z is exactly the size of the reference window X
            if len(self.X_prime) == self.n:
                print(f"\n  [Classification Phase @ Time {self.t}]")
                mah_distances = []
                
                # Compare new distribution Z against all historical distributions X_i
                for i, regime in enumerate(self.regimes):
                    dist_sq = self._compute_kernel_mahalanobis_distance(self.X_prime, regime)
                    mah_distances.append(dist_sq)
                    print(f"    - vs Regime {i}: KMD^2 = {dist_sq:.4f}")
                
                if mah_distances:
                    best_i = np.argmin(mah_distances)
                    min_dist = mah_distances[best_i]
                else:
                    best_i = -1
                    min_dist = float('inf')
                        
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
                    new_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {new_id} (Min KMD^2 = {min_dist:.4f} > {self.nu})\n")
                    self.active_regime_id = new_id
                    self.regimes.append(new_regime_dict)
                
                # Restart Active Monitoring
                self.X = list(self.X_prime)
                self.active_W = W_u.copy()
                self.Y = []
                self.state = "WARMUP_Y"
                
        self.pearson_div_scores.append(score)
        return score