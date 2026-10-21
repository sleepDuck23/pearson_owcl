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


def inverse_gram(K, reg_lambda):
    """Gram inverse for an empirical operator averaged over n samples."""
    eye = np.eye(len(K))
    W = np.linalg.solve(K + len(K) * reg_lambda * eye, eye)
    return (W + W.T) / 2


def slide_inverse(model, incoming):
    """Update (K + n*lambda*I)^-1; periodically rebuild to limit drift."""
    X_new = model.X[1:] + [incoming]

    def rebuild():
        return inverse_gram(model._rbf_kernel(X_new, X_new), model.reg_lambda)

    if model.t % max(model.n, 50) == 0:
        return rebuild()
    W = model.active_W
    if not np.isfinite(W[0, 0]) or W[0, 0] <= 0:
        return rebuild()
    reduced = W[1:, 1:] - np.outer(W[1:, 0], W[0, 1:]) / W[0, 0]
    k = model._rbf_kernel(model.X[1:], incoming).ravel()
    v = np.dot(reduced, k)
    diagonal = float(model._rbf_kernel(incoming, incoming)[0, 0])
    schur = diagonal + model.n * model.reg_lambda - np.dot(k, v)
    if not np.isfinite(schur) or schur < 0.5 * model.n * model.reg_lambda:
        return rebuild()
    updated = np.block([[reduced + np.outer(v, v) / schur, -v[:, None] / schur],
                        [-v[None, :] / schur, np.array([[1 / schur]])]])
    return (updated + updated.T) / 2


def operator_distance(mmd_squared, delta, W, reg_lambda):
    """<Delta, (T + lambda I)^-1 Delta>, using the Woodbury identity.

    W=(K+n*lambda*I)^-1 and delta=Phi*Delta. For centered T, use
    K=H K_XX H and delta=H Phi*Delta. Retain the full ||Delta||^2 term
    to include directions outside the reference feature span.
    """
    value = float((mmd_squared - np.dot(delta, np.dot(W, delta))) / reg_lambda)
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite regularized RKHS distance")
    return max(0.0, value)


def pearson_score(model, Z, X, W, mu_K, gamma, h=None):
    """Uncentered Pearson score: 1/2 <Delta, (T_X + lambda I)^-1 Delta>."""
    if h is None:
        h = np.mean(model._rbf_kernel(X, Z, gamma_override=gamma), axis=1)
    mmd_squared = (np.mean(mu_K)
                   + np.mean(model._rbf_kernel(Z, Z, gamma_override=gamma))
                   - 2 * np.mean(h))
    return 0.5 * operator_distance(mmd_squared, h - mu_K, W, model.reg_lambda)


class PearsonOWRCPD:
    def __init__(self, n, m, tau, nu, k_conf, reg_lambda=1e-3, gamma=0.1, factor=1.0, dynamic_gamma=True, normalize_distance=False):
        """
        Open-World Change-Point Detection with Recurrent Regime Recall.
        - Detection: Amortized O(n^2 + m^2) recursive Pearson Divergence (regularized RKHS second-moment operator).
        - Classification: Static Pearson Divergence (regularized RKHS Pearson-type discrepancy).
        - Kernel: Optional Dynamic Median Heuristic bandwidth tuning per regime.
        - Distance: Optional Hypersphere Normalization for scale invariance.

        Detection and classification use the uncentered empirical second
        moment T_X = sum_j phi(X_j) tensor phi(X_j) / n. reg_lambda is
        lambda in T_X + lambda I, so the Gram ridge is n * reg_lambda.
        """
        validate_parameters(n, m, k_conf, gamma, factor, reg_lambda)
        self.n = n                  # Reference window size (and Classification window size)
        self.m = m                  # Analysis window size
        self.tau = tau              # Detection threshold (Pearson Divergence)
        self.nu = nu                # Novelty threshold (Maximum allowed Pearson Divergence for a match)
        self.k_conf = k_conf        # Consecutive windows to confirm change
        self.reg_lambda = reg_lambda 
        self.gamma = gamma          # Initial gamma (remains static if dynamic_gamma=False)
        self.factor = factor        # Optional scaling factor 
        self.dynamic_gamma = dynamic_gamma # Boolean flag to toggle dynamic gamma computation
        self.normalize_distance = normalize_distance # Boolean flag to toggle hypersphere projection
        
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
        self.active_mu_K = np.array([])
        
        # Regimes Library: List of dicts {'X': array, 'W': array, 'mu_K': array, 'gamma': float}
        self.regimes = []       
        self.active_regime_id = 0
        
        # Logs
        self.global_changepoints = []
        self.pearson_div_scores = []
        self.regime_log = []

    def _sq_distances_v0(self, X, Y):
        return squared_distances(X, Y, normalize=False)

    def _sq_distances(self, X, Y):
        return squared_distances(X, Y, normalize=True)

    def _get_active_distances(self, X, Y):
        """Routes to the appropriate distance metric based on the normalization toggle."""
        if self.normalize_distance:
            return self._sq_distances(X, Y)
        else:
            return self._sq_distances_v0(X, Y)

    def _rbf_kernel(self, X, Y, gamma_override=None):
        """Gaussian RBF kernel matrix computation."""
        dist_sq = self._get_active_distances(X, Y)
        g = gamma_override if gamma_override is not None else self.gamma
        return np.exp(-g * dist_sq)

    def _compute_full_W_and_update_gamma(self, X_data):
        """
        1. Computes squared distances using the toggled metric.
        2. Applies the Median Heuristic to tune gamma (if dynamic_gamma=True).
        3. Computes exact inverse metric tensor W and reference mean mu_K.
        """
        X_arr = np.array(X_data)
        dist_sq = self._get_active_distances(X_arr, X_arr)
        
        # Only compute and update via median heuristic if the flag is True
        if self.dynamic_gamma:
            self.gamma = median_gamma(dist_sq, self.factor, self.gamma)

        # Compute K_XX using the active gamma (dynamically tuned or static)
        K_XX = np.exp(-self.gamma * dist_sq)
        
        # Compute exact inverse metric tensor
        W = inverse_gram(K_XX, self.reg_lambda)

        # Compute the reference mean embedding vector
        mu_K = np.mean(K_XX, axis=1)

        return W, mu_K, self.gamma

    def _compute_pearson_divergence(self, Z, regime_dict):
        """
        Compute 1/2 <mu_Z-mu_X, (T_X + lambda I)^-1 (mu_Z-mu_X)>.
        T_X is the uncentered reference second moment, averaged over X.
        """
        X_i = regime_dict['X']
        W_i = regime_dict['W']
        mu_K_i = regime_dict['mu_K']
        gamma_i = regime_dict['gamma']
        
        return pearson_score(self, Z, X_i, W_i, mu_K_i, gamma_i)

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
                W_init, mu_K_init, gamma_init = self._compute_full_W_and_update_gamma(self.X)
                self.regimes.append({
                    'X': list(self.X), 
                    'W': W_init, 
                    'mu_K': mu_K_init, 
                    'gamma': gamma_init
                })
                
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
                
        # --- PHASE 3: Active Monitoring (Recursive Sliding Window) ---
        elif self.state == "ACTIVE":
            x_t = x                 # Entering Y
            x_t_m = self.Y[0]       # Leaving Y, entering X
            
            # W = (K_XX + n * lambda I)^-1, with fixed gamma.
            self.active_W = slide_inverse(self, x_t_m)
            X_reduced = self.X[1:]
            k_in = self._rbf_kernel(X_reduced, x_t_m).flatten()

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
            
            # 4. Evaluate regularized RKHS Pearson-type discrepancy
            # Dynamically recalculate the sliding reference mean
            self.active_mu_K = np.mean(self._rbf_kernel(self.X, self.X), axis=1)
            
            score = pearson_score(
                self, self.Y, self.X, self.active_W, self.active_mu_K,
                self.gamma, h=self.active_h)
            
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
                        
                new_W, new_mu_K, new_gamma = self._compute_full_W_and_update_gamma(self.X_prime) 
                
                if min_div <= self.nu:
                    print(f"  -> RESULT: Assigned to KNOWN Regime {best_i} (Min Div = {min_div:.4f} <= {self.nu})\n")
                    self.active_regime_id = best_i
                    self.regimes[best_i] = {
                        'X': list(self.X_prime), 
                        'W': new_W, 
                        'mu_K': new_mu_K, 
                        'gamma': new_gamma
                    }
                else:
                    new_id = len(self.regimes)
                    print(f"  -> RESULT: Initiating NOVEL Regime {new_id} (Min Div = {min_div:.4f} > {self.nu})\n")
                    self.active_regime_id = new_id
                    self.regimes.append({
                        'X': list(self.X_prime), 
                        'W': new_W, 
                        'mu_K': new_mu_K, 
                        'gamma': new_gamma
                    })
                
                self.X = list(self.X_prime)
                self.active_W = new_W.copy()
                self.active_mu_K = new_mu_K.copy()
                self.Y = []
                self.state = "WARMUP_Y"
                
        self.pearson_div_scores.append(score)
        return score
