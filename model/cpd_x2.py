import numpy as np

class Pearson_CPD:
    def __init__(self, n, m, tau, reg_lambda=1e-3, gamma=0.1):
        """
        Optimized Kernelized Pearson (chi-squared) Divergence Change-Point Detection.
        Utilizes O(N^2) recursive Schur complement updates for continuous sliding windows.
        """
        self.n = n                  # Reference window size
        self.m = m                  # Test window size
        self.tau = tau              # Pearson divergence threshold for CPD
        self.reg_lambda = reg_lambda # Tikhonov regularization (lambda)
        self.gamma = gamma          # RBF kernel bandwidth parameter
        
        print(f"--- Model Initialized: Optimized Pearson CPD ---")
        print(f" > Reference (n) / Test (m) : {self.n} / {self.m}")
        print(f" > Detection Threshold (tau): {self.tau:.4f}")
        print(f" > Regularization (lambda)  : {self.reg_lambda}")
        print(f" > RBF Kernel Bandwidth     : {self.gamma}")
        
        # Buffers and state
        self.state = "INIT_Y" 
        self.t = 0
        self.X = []
        self.Y = []
        
        # Metric Tensor and cross-kernel vector
        self.W = np.array([])
        self.h = np.array([])
        
        # Logging
        self.global_changepoints = []
        self.pearson_scores = []

    def _rbf_kernel(self, X, Y):
        """Pure NumPy broadcasting for the RBF kernel."""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        X2 = np.sum(X**2, axis=1).reshape(-1, 1)
        Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
        return np.exp(-self.gamma * (X2 + Y2 - 2 * np.dot(X, Y.T)))

    def _initialize_tensors(self):
        """Computes the exact initial W and h in O(N^3) once at warmup."""
        X_arr = np.array(self.X)
        Y_arr = np.array(self.Y)
        
        # K_XX + lambda * I
        K_XX = self._rbf_kernel(X_arr, X_arr)
        self.W = np.linalg.inv(K_XX + self.reg_lambda * np.eye(self.n))
        
        # Cross kernel mean
        self.h = np.mean(self._rbf_kernel(X_arr, Y_arr), axis=1)

    def update(self, x):
        """Main online processing step."""
        self.t += 1
        x = np.atleast_1d(x)
        score = np.nan
        
        # --- PHASE 1: Fill Test Window (Y) ---
        if self.state == "INIT_Y":
            self.Y.append(x)
            if len(self.Y) == self.m:
                self.state = "INIT_X"
                
        # --- PHASE 2: Fill Reference Window (X) ---
        elif self.state == "INIT_X":
            x_out_Y = self.Y.pop(0)
            self.X.append(x_out_Y)
            self.Y.append(x)
            
            if len(self.X) == self.n:
                self._initialize_tensors()
                self.state = "ACTIVE"
                
        # --- PHASE 3: Active Monitoring (O(N^2) Updates) ---
        elif self.state == "ACTIVE":
            x_t = x                 # Entering Y
            x_t_m = self.Y[0]       # Leaving Y, entering X
            # Note: x_t_n_m (leaving X) is simply self.X[0]
            
            # 1. Downdate W (Remove oldest reference sample)
            w11 = self.W[0, 0]
            w12 = self.W[0, 1:]
            w21 = self.W[1:, 0]
            W22 = self.W[1:, 1:]
            
            W_reduced = W22 - np.outer(w21, w12) / w11
            
            # 2. Update W (Add incoming reference sample)
            X_reduced = self.X[1:]  
            k_in = self._rbf_kernel(X_reduced, x_t_m).flatten()
            
            v = W_reduced @ k_in
            # (1.0 + lambda) represents the regularized self-kernel evaluation
            schur_scalar = (1.0 + self.reg_lambda) - np.dot(k_in, v)
            schur_scalar = max(1e-12, schur_scalar) # Numerical safety
            beta = 1.0 / schur_scalar
            
            # Reconstruct blocked matrix
            top_left = W_reduced + beta * np.outer(v, v)
            top_right = -beta * v[:, None]
            bottom_left = -beta * v[None, :]
            bottom_right = np.array([[beta]])
            
            self.W = np.block([
                [top_left,    top_right],
                [bottom_left, bottom_right]
            ])
            
            # 3. Update h vector recursively
            h_reduced = self.h[1:] 
            k_in_xt = self._rbf_kernel(X_reduced, x_t).flatten()
            
            # Shift the mean for the remaining samples
            h_reduced += (1.0 / self.m) * (k_in_xt - k_in)
            
            # Physically update the buffers
            self.Y.pop(0)
            self.Y.append(x_t)
            self.X.pop(0)
            self.X.append(x_t_m)
            
            # Compute the cross-kernel mean for the brand new reference sample
            h_new = np.mean(self._rbf_kernel(x_t_m, self.Y).flatten())
            self.h = np.append(h_reduced, h_new)
            
            # 4. Evaluate Divergence Score
            theta = self.W @ self.h
            score = np.dot(theta, self.h) - 1.0
            
            if abs(score) > self.tau:
                print(f"\nTime {self.t}: *** CHANGE DETECTED *** (Pearson Score={score:.4f} > {self.tau:.4f})")
                self.global_changepoints.append(self.t)
                
                # Flush buffers and restart initialization
                self.X = []
                self.Y = []
                self.W = np.array([])
                self.h = np.array([])
                self.state = "INIT_Y"

        self.pearson_scores.append(score)
        return score