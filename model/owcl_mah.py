import numpy as np

class Mahalanobis_OWR_CPD:
    def __init__(self, n, m, tau, nu, k_conf, reg_lambda=1e-3, gamma=0.1):
        """
        Open-World Change-Point Detection with Kernel Mahalanobis Classification.
        - Detection: O(N^2) recursive Pearson Divergence (Kernel Space).
        - Classification: Kernel Mahalanobis Distance derived from the W tensor.
        """
        self.n = n                  # Reference window size
        self.m = m                  # Analysis window size
        self.tau = tau              # Detection threshold (Pearson Divergence)
        self.nu = nu                # Novelty threshold (Kernel Mahalanobis Distance - MAX allowed distance)
        self.k_conf = k_conf        # Consecutive windows to confirm change
        self.reg_lambda = reg_lambda # Tikhonov regularization
        self.gamma = gamma          # RBF kernel bandwidth
        
        # System State
        self.t = 0
        self.state = "INIT_REGIME" 
        self.consec = 0
        
        # Buffers
        self.X = []                 # Active reference window (slides continuously)
        self.Y = []                 # Active analysis window (slides continuously)
        self.X_prime = []           # Pure representative window after change
        
        # Active Monitoring Tensors (Updated in O(N^2))
        self.active_W = np.array([])
        self.active_h = np.array([])
        
        # Regimes Library: List of dicts {'X': array, 'W': array}
        self.regimes = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints = []
        self.pearson_div_scores = []
        self.regime_log = []

    def _rbf_kernel(self, X, Y):
        """Gaussian RBF kernel matrix computation."""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        X2 = np.sum(X**2, axis=1).reshape(-1, 1)
        Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
        return np.exp(-self.gamma * (X2 + Y2 - 2 * np.dot(X, Y.T)))

    def _compute_full_W(self, X_data):
        """Computes the exact inverse metric tensor in O(N^3) (used for initialization)."""
        X_arr = np.array(X_data)
        K_XX = self._rbf_kernel(X_arr, X_arr)
        return np.linalg.inv(K_XX + self.reg_lambda * np.eye(len(X_arr)))

    def _compute_kernel_mahalanobis_distance(self, Z, regime_dict):
        """
        Computes Kernel Mahalanobis Distance using the existing W tensor and cross-kernel vector.
        Distance = abs(1.0 - (h^T W h)), where h is the cross-kernel mean vector.
        """
        X_i = regime_dict['X']
        W_i = regime_dict['W']
        
        # Cross-kernel vector between stored historical regime and the new pure window Z
        h_iZ = np.mean(self._rbf_kernel(X_i, Z), axis=1)
        
        # Kernel Mahalanobis overlap (similarity score)
        overlap = np.dot(W_i @ h_iZ, h_iZ)
        
        # Convert to a distance metric (0.0 = identical match, higher = disjoint)
        distance = abs(1.0 - overlap)
        return float(distance)

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
                W_init = self._compute_full_W(self.X)
                
                # Store regime using only X and W (no raw covariance needed)
                self.regimes.append({'X': list(self.X), 'W': W_init})
                
                self.active_W = W_init.copy()
                self.active_regime_id = 0
                self.state = "WARMUP_Y"
                
        # --- PHASE 2: Fill Analysis Window ---
        elif self.state == "WARMUP_Y":
            self.Y.append(x)
            self.regime_log.append(self.active_regime_id)
            
            if len(self.Y) == self.m:
                self.active_h = np.mean(self._rbf_kernel(self.X, self.Y), axis=1)
                self.state = "ACTIVE"
                
        # --- PHASE 3: Active Monitoring (O(N^2) Sliding Window) ---
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
            
            # 4. Evaluate Divergence Score
            overlap = np.dot(self.active_W @ self.active_h, self.active_h)
            score = abs(overlap - 1.0) 
            
            if score > self.tau:
                self.consec += 1
            else:
                self.consec = 0
                
            if self.consec >= self.k_conf:
                cp_time = self.t
                self.global_changepoints.append(cp_time)
                print(f"\nTime {cp_time} (detected at {self.t}): *** CHANGE CONFIRMED *** (Div={score:.4f} > {self.tau})")
                
                self.consec = 0
                self.X_prime = []
                self.state = "COLLECT_PURE_WINDOW"
                
        # --- PHASE 4: OWR Classification via Kernel Mahalanobis Distance ---
        elif self.state == "COLLECT_PURE_WINDOW":
            self.X_prime.append(x)
            self.regime_log.append(-1)
            
            if len(self.X_prime) == self.n:
                print(f"\n  [Classification Phase @ Time {self.t}]")
                mah_distances = []
                
                # Evaluate Kernel Mahalanobis distance against all stored regimes
                for i, regime in enumerate(self.regimes):
                    dist = self._compute_kernel_mahalanobis_distance(self.X_prime, regime)
                    mah_distances.append(dist)
                    print(f"    - vs Regime {i}: Kernel Mahalanobis Dist = {dist:.4f}")
                
                if mah_distances:
                    best_i = np.argmin(mah_distances)
                    min_dist = mah_distances[best_i]
                else:
                    best_i = -1
                    min_dist = float('inf')
                        
                # Pre-calculate W for the new pure window
                new_W = self._compute_full_W(self.X_prime)
                
                # Classification Decision (Distance check: smaller distance is better)
                if min_dist <= self.nu:
                    print(f"  -> RESULT: Assigned to KNOWN Regime {best_i} (Min Dist = {min_dist:.4f} <= {self.nu})\n")
                    self.active_regime_id = best_i
                    self.regimes[best_i] = {'X': list(self.X_prime), 'W': new_W}
                else:
                    new_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {new_id} (Min Dist = {min_dist:.4f} > {self.nu})\n")
                    self.active_regime_id = new_id
                    self.regimes.append({'X': list(self.X_prime), 'W': new_W})
                
                # Restart Active Monitoring using the newly verified space
                self.X = list(self.X_prime)
                self.active_W = new_W.copy()
                self.Y = []
                self.state = "WARMUP_Y"
                
        self.pearson_div_scores.append(score)
        return score