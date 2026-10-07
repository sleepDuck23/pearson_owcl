import numpy as np

class Pearson_OWR_CPD:
    def __init__(self, n, m, tau, nu, k_conf, reg_lambda=1e-3, gamma=0.1):
        """
        Open-World Change-Point Detection with Recurrent Regime Recall.
        - Detection: O(N^2) recursive Pearson Divergence.
        - Classification: Static Pearson Divergence evaluated against historical baselines.
        - Kernel: Dynamic Median Heuristic bandwidth tuning per regime.
        """
        self.n = n                  # Reference window size (and Classification window size)
        self.m = m                  # Analysis window size
        self.tau = tau              # Detection threshold (Pearson Divergence)
        self.nu = nu                # Novelty threshold (Maximum allowed Pearson Divergence for a match)
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
        
        # Regimes Library: List of dicts {'X': array, 'W': array, 'gamma': float}
        self.regimes = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints = []
        self.pearson_div_scores = []
        self.regime_log = []

    def _sq_distances_v0(self, X, Y):
        """Computes the squared Euclidean distance matrix."""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        X2 = np.sum(X**2, axis=1).reshape(-1, 1)
        Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
        dist_sq = X2 + Y2 - 2 * np.dot(X, Y.T)
        return np.maximum(dist_sq, 0.0)

    def _sq_distances(self, X, Y):
        """
        Computes scale-invariant squared Euclidean distance by 
        internally projecting data onto a unit hypersphere.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        
        # 1. Internal Normalization (makes the model scale-invariant)
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        
        # 2. Compute distance in Cosine space
        # Since vectors are unit length, X^2 + Y^2 is exactly 2.0
        dist_sq = 2.0 - 2.0 * np.dot(X_norm, Y_norm.T)
        
        return np.maximum(dist_sq, 0.0)

    def _rbf_kernel(self, X, Y, gamma_override=None):
        """Gaussian RBF kernel matrix computation."""
        dist_sq = self._sq_distances(X, Y)
        g = gamma_override if gamma_override is not None else self.gamma
        return np.exp(-g * dist_sq)

    def _compute_full_W_and_update_gamma(self, X_data):
        """
        1. Computes squared distances.
        2. Applies the Median Heuristic to tune gamma.
        3. Computes exact inverse metric tensor W in O(N^3).
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
            
        self.gamma = 2.5 * (1.0 / (2.0 * median_sq_dist))
        
        # Compute K_XX using the newly tuned gamma
        K_XX = np.exp(-self.gamma * dist_sq)
        
        # Compute exact inverse metric tensor
        W = np.linalg.inv(K_XX + self.reg_lambda * np.eye(len(X_arr)))

        # Diagnostic prints
        print(f"  [Debug] K_XX Condition Number: {np.linalg.cond(K_XX):.2e}")
        print(f"  [Debug] Distance Variance: {np.var(pairwise_sq_dists):.4f}")
        print(f"  [Debug] K_XX Mean Off-Diagonal: {np.mean(K_XX[np.triu_indices_from(K_XX, k=1)]):.4f}")
        
        return W, self.gamma

    def _compute_pearson_divergence(self, Z, regime_dict):
        """
        Computes the absolute Pearson Divergence from the baseline of 1.0.
        Crucially uses the historical regime's specific gamma to project 
        the new data correctly into that regime's feature space.
        """
        X_i = regime_dict['X']
        W_i = regime_dict['W']
        gamma_i = regime_dict['gamma']
        
        # Cross-kernel vector using historical gamma
        h_iZ = np.mean(self._rbf_kernel(X_i, Z, gamma_override=gamma_i), axis=1)
        
        # Pearson overlap score
        overlap = np.dot(W_i @ h_iZ, h_iZ)
        
        # Convert to strict absolute divergence metric
        return abs(overlap - 1.0)

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
                # Dynamically tune gamma and compute tensor
                W_init, gamma_init = self._compute_full_W_and_update_gamma(self.X)
                self.regimes.append({'X': list(self.X), 'W': W_init, 'gamma': gamma_init})
                
                # Clone into active monitoring arrays
                self.active_W = W_init.copy()
                self.active_regime_id = 0
                self.state = "WARMUP_Y"
                
        # --- PHASE 2: Fill Analysis Window ---
        elif self.state == "WARMUP_Y":
            self.Y.append(x)
            self.regime_log.append(self.active_regime_id)
            
            if len(self.Y) == self.m:
                # Initialize active cross-kernel vector
                self.active_h = np.mean(self._rbf_kernel(self.X, self.Y), axis=1)
                self.state = "ACTIVE"
                
        # --- PHASE 3: Active Monitoring (O(N^2) Sliding Window) ---
        elif self.state == "ACTIVE":
            x_t = x                 # Entering Y
            x_t_m = self.Y[0]       # Leaving Y, entering X
            
            # 1. Downdate W (Remove oldest reference sample)
            w11 = self.active_W[0, 0]
            w12 = self.active_W[0, 1:]
            w21 = self.active_W[1:, 0]
            W22 = self.active_W[1:, 1:]
            
            W_reduced = W22 - np.outer(w21, w12) / w11
            
            # 2. Update W (Add incoming reference sample)
            X_reduced = self.X[1:]  
            k_in = self._rbf_kernel(X_reduced, x_t_m).flatten()
            
            v = W_reduced @ k_in
            schur_scalar = max(1e-12, (1.0 + self.reg_lambda) - np.dot(k_in, v))
            beta = 1.0 / schur_scalar
            
            self.active_W = np.block([
                [W_reduced + beta * np.outer(v, v), -beta * v[:, None]],
                [-beta * v[None, :],                np.array([[beta]])]
            ])
            
            # 3. Update h vector recursively
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
                
        # --- PHASE 4: OWR Classification ---
        elif self.state == "COLLECT_PURE_WINDOW":
            self.X_prime.append(x)
            self.regime_log.append(-1)
            
            if len(self.X_prime) == self.n:
                print(f"\n  [Classification Phase @ Time {self.t}]")
                div_scores = []
                
                # Check Divergence against all stored regimes (Lower is better)
                for i, regime in enumerate(self.regimes):
                    div = self._compute_pearson_divergence(self.X_prime, regime)
                    div_scores.append(div)
                    print(f"    - vs Regime {i}: Pearson Divergence = {div:.4f}")
                
                if div_scores:
                    best_i = np.argmin(div_scores)
                    min_div = div_scores[best_i]
                else:
                    best_i = -1
                    min_div = float('inf')
                        
                # 1. Profile the new window & extract the newly tuned gamma
                new_W, new_gamma = self._compute_full_W_and_update_gamma(self.X_prime) 
                
                # 2. Classification Decision
                if min_div <= self.nu:
                    print(f"  -> RESULT: Assigned to KNOWN Regime {best_i} (Min Div = {min_div:.4f} <= {self.nu})\n")
                    self.active_regime_id = best_i
                    
                    # Overwrite historical regime baseline with the new, shifted data
                    self.regimes[best_i] = {'X': list(self.X_prime), 'W': new_W, 'gamma': new_gamma}
                else:
                    new_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {new_id} (Min Div = {min_div:.4f} > {self.nu})\n")
                    self.active_regime_id = new_id
                    
                    self.regimes.append({'X': list(self.X_prime), 'W': new_W, 'gamma': new_gamma})
                
                # Restart Active Monitoring using the newly verified space
                self.X = list(self.X_prime)
                self.active_W = new_W.copy()
                self.Y = []
                self.state = "WARMUP_Y"
                
        self.pearson_div_scores.append(score)
        return score