import numpy as np

class DictionaryOWRCPD:
    def __init__(self, n=150, m=50, tau=0.35, nu=0.08, k_conf=6, gamma=0.6, 
                 reg_lambda=0.1, dict_max_size=100, coherence_thresh=0.99, ald_thresh=0.01, dynamic_gamma=False):
        # Base CPD and OWR parameters
        self.n = n
        self.m = m
        self.tau = tau
        self.nu = nu
        self.k_conf = k_conf
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.dynamic_gamma = dynamic_gamma
        
        # Dictionary extraction parameters
        self.D_max = dict_max_size
        self.coherence_thresh = coherence_thresh
        self.ald_thresh = ald_thresh
        
        # State tracking
        self.t = 0
        self.in_warmup = True
        self.warmup_buffer = []
        self.test_window = []
        self.consec_alarms = 0
        
        # Dictionary memory
        self.dictionary = []
        self.K_inv = np.empty((0, 0)) # Used for ALD extraction
        
        # Global tracking variables
        self.global_changepoints = []
        self.regime_log = []
        self.regimes = []          # Stores the dictionaries of known regimes
        self.current_regime = None

    def _rbf_kernel(self, X, Y):
        """Computes the RBF kernel matrix between two sets of vectors."""
        X = np.array(X).reshape(len(X), -1)
        Y = np.array(Y).reshape(len(Y), -1)
        dist_sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * np.maximum(dist_sq, 0.0))

    def _add_to_dictionary(self, x_t):
        """Evaluates a single point for dictionary inclusion using Coherence and ALD."""
        x_t = np.atleast_2d(x_t)
        
        if len(self.dictionary) == 0:
            self.dictionary.append(x_t.flatten())
            self.K_inv = np.array([[1.0]])
            return True

        # 1. Coherence Criterion
        k_t = self._rbf_kernel(x_t, self.dictionary).flatten()
        max_coherence = np.max(np.abs(k_t))
        
        if max_coherence > self.coherence_thresh:
            return False 

        # 2. Approximate Linear Dependency (ALD) Criterion
        delta_t = 1.0 - np.dot(k_t, self.K_inv @ k_t)
        
        if delta_t > self.ald_thresh and len(self.dictionary) < self.D_max:
            self.dictionary.append(x_t.flatten())
            a_t = self.K_inv @ k_t
            inv_delta = 1.0 / max(delta_t, 1e-12)
            
            # Fast incremental Schur complement update (O(D^2))
            new_K_inv = np.zeros((len(self.dictionary), len(self.dictionary)))
            new_K_inv[:-1, :-1] = self.K_inv + inv_delta * np.outer(a_t, a_t)
            new_K_inv[:-1, -1] = -inv_delta * a_t
            new_K_inv[-1, :-1] = -inv_delta * a_t
            new_K_inv[-1, -1] = inv_delta
            
            self.K_inv = new_K_inv
            return True
        return False

    def _compute_pearson_divergence(self, Y):
        """
        DETECTION PHASE: Computes Pearson Squared Divergence using the sparse dictionary 
        instead of the dense reference window. Complexity drops from O(N^3) to O(D^3).
        """
        D = np.array(self.dictionary)
        Y_arr = np.array(Y)
        
        K_DD = self._rbf_kernel(D, D)
        K_DY = self._rbf_kernel(D, Y_arr)
        K_YY = self._rbf_kernel(Y_arr, Y_arr)
        
        # W_D operates on a small DxD matrix, completely avoiding the overflows of NxN
        W_D = np.linalg.inv(K_DD + self.reg_lambda * np.eye(len(D)))
        h_D = np.mean(K_DY, axis=1, keepdims=True)
        
        # Solve for density ratio coefficients over the dictionary basis
        theta = W_D @ h_D
        
        # Pearson Divergence
        pe_div = np.mean(K_YY) - float(np.dot(h_D.T, theta))
        return max(pe_div, 0.0)

    def _compute_mahalanobis(self, dict_known, dict_active):
        """
        CLASSIFICATION PHASE: Computes the Kernel Mahalanobis Distance between the 
        centers of two sparse dictionaries in the RKHS.
        """
        A = np.array(dict_known)
        B = np.array(dict_active)
        
        K_AA = self._rbf_kernel(A, A)
        K_AB = self._rbf_kernel(A, B)
        
        d = len(A)
        # Centering matrix for the known dictionary
        H = np.eye(d) - np.ones((d, d)) / d
        
        # Centered Covariance and Precision Matrix in RKHS
        K_c = H @ K_AA @ H
        W_c = np.linalg.inv(K_c + self.reg_lambda * np.eye(d))
        
        # Difference in mean embeddings
        h_tilde = np.mean(K_AB, axis=1, keepdims=True) - np.mean(K_AA, axis=1, keepdims=True)
        
        # Mahalanobis distance computation
        mah_dist = float(h_tilde.T @ W_c @ h_tilde)
        return max(mah_dist, 0.0)

    def _assign_regime(self):
        """Classifies the active dictionary using Kernel Mahalanobis Distance."""
        if not self.regimes:
            self.current_regime = 0
            self.regimes.append(self.dictionary)
            return

        mah_distances = []
        for known_dict in self.regimes:
            dist = self._compute_mahalanobis(known_dict, self.dictionary)
            mah_distances.append(dist)
            
        best_match_idx = np.argmin(mah_distances)
        min_dist = mah_distances[best_match_idx]

        # Mahalanobis identifies similar regimes with distances below the threshold nu
        if min_dist <= self.nu:
            self.current_regime = best_match_idx
            self.regimes[best_match_idx] = self.dictionary 
        else:
            self.current_regime = len(self.regimes)
            self.regimes.append(self.dictionary)

    def update(self, x):
        """Processes a single streaming data point."""
        self.t += 1
        
        if self.in_warmup:
            self.warmup_buffer.append(x)
            self.regime_log.append(-1) 
            
            if len(self.warmup_buffer) == self.n:
                # Dynamic Gamma Lock on Warmup Buffer
                if self.dynamic_gamma:
                    distances = []
                    buf_arr = np.array(self.warmup_buffer).reshape(-1, 1)
                    for i in range(len(buf_arr)):
                        for j in range(i+1, len(buf_arr)):
                            distances.append((buf_arr[i] - buf_arr[j])**2)
                    safe_median = max(np.median(distances), 1e-5)
                    raw_gamma = 1.0 / (2.0 * safe_median)
                    self.gamma = np.clip(raw_gamma, a_min=0.01, a_max=10.0)

                # Extract sparse dictionary
                self.dictionary = []
                self.K_inv = np.empty((0, 0))
                for pt in self.warmup_buffer:
                    self._add_to_dictionary(pt)
                
                # Classify regime via Mahalanobis
                self._assign_regime()
                
                self.in_warmup = False
                self.test_window = []
                self.consec_alarms = 0
                
            return 0.0

        # --- Detection Phase ---
        self.test_window.append(x)
        if len(self.test_window) > self.m:
            self.test_window.pop(0)
            
        if len(self.test_window) == self.m:
            # Detect using Pearson Divergence on the Dictionary
            score = self._compute_pearson_divergence(self.test_window)
            
            if score > self.tau:
                self.consec_alarms += 1
                if self.consec_alarms >= self.k_conf:
                    change_time = self.t 
                    self.global_changepoints.append(change_time)
                    
                    self.in_warmup = True
                    self.warmup_buffer = []
                    self.regime_log.append(-1)
                    return score
            else:
                self.consec_alarms = 0
                
            self.regime_log.append(self.current_regime)
            return score
            
        self.regime_log.append(-1) 
        return 0.0