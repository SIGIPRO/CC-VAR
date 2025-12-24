import numpy as np
import copy

class CCVAR:
    def __init__(self, algorithmParam, cellularComplex, theta_initializer=None):
        self.__algorithm_parameter_setup(algorithmParam)
        
        # Generic keys: [0, 1, 2...]
        self._data_keys = sorted([i for i, en in enumerate(self._data_enabler) if en])
        
        self._phi = dict()
        self._r = dict()
        self._theta = dict()
        
        # 1. Generic Topology Construction
        self._construct_laplacian(cellularComplex)
        
        # 2. Data Buffers
        self._data_initializer()
        
        # 3. Weights
        if theta_initializer is None:
            self._theta_initializer = self.__zero_initializer
        else:
            self._theta_initializer = theta_initializer

        self._allocate_state_variables()

    def set_topology(self, cellularComplex):
        """Public method for MAB/Coordinator to update topology."""
        self._construct_laplacian(cellularComplex)
        self._handle_topology_change(cellularComplex)

    def _handle_topology_change(self, cellularComplex):
        """Virtual method for state handling during topology change."""
        self._allocate_state_variables()
        self._data_initializer()

    def update(self, inputData):
        """Online Learning Loop (Single Step)."""
        featureDict = self._feature_gen()
        # import pdb; pdb.set_trace()
        for key in self._data_keys:
            if key not in inputData: continue
            
            S = featureDict[key]
            # Ensure target is (N, 1)
            target = inputData[key].reshape(-1, 1)
            
            # --- Optimization Hooks ---
            self._update_state(key, S, target)
            eta = self._compute_step_size(key)
            self._apply_descent_step(key, eta)

             # NEW (MATLAB Equivalent):
            old_data = self._data[key][:, 1:]
            
            # current_step_pred[k] is flat (N,), make it (N, 1)
            new_col = inputData[key].reshape(-1, 1)
            
            self._data[key] = np.hstack([old_data, new_col])

    def forecast(self, steps):
        """
        Recursive N-step forecasting with Self-Adaptation.
        Matches MATLAB behavior: Updates weights based on predictions during the window.
        """
        preds = {k: np.zeros((self._N[k], steps)) for k in self._data_keys}
        
        # 1. Snapshot EVERYTHING (Data, Norms, AND Optimizer State)
        saved_data = copy.deepcopy(self._data)
        saved_norm = copy.deepcopy(self._norm_scale)
        saved_theta = copy.deepcopy(self._theta)
        saved_phi = copy.deepcopy(self._phi)
        saved_r = copy.deepcopy(self._r)
        
        # Reset norm scale for prediction phase (Matches MATLAB logic)
        for k in self._data_keys:
            self._norm_scale[k] = 0

        for t in range(steps):
            # Generate features (Uses current temporary buffer)
            feats = self._feature_gen()
            
            current_step_pred = {}
            for k in self._data_keys:
                if self._theta[k] is None: continue
                
                # A. Predict
                S = feats[k]
                y_pred = S @ self._theta[k]
                
                preds[k][:, t] = y_pred.flatten()
                current_step_pred[k] = y_pred.flatten()

                         # B. NO UPDATE HERE! 
                # We skip _update_state, _compute_step_size, and _apply_descent_step.
                # We trust the weights learned from real data.
                
                # # B. Self-Adaptation (Crucial for T > 1)
                # We treat the prediction as the ground truth for a temporary weight update.
                self._update_state(k, S, y_pred) 
                eta = self._compute_step_size(k)
                self._apply_descent_step(k, eta)
            
            # C. Shift Buffer for next step
            if t < steps - 1:
                for k in self._data_keys:
                     # NEW (MATLAB Equivalent):
                    old_data = self._data[k][:, 1:]
                    
                    # current_step_pred[k] is flat (N,), make it (N, 1)
                    new_col = current_step_pred[k].reshape(-1, 1)
                    
                    self._data[k] = np.hstack([old_data, new_col])

        # 2. Restore State to pre-forecast values
        self._data = saved_data
        self._norm_scale = saved_norm
        self._theta = saved_theta
        self._phi = saved_phi
        self._r = saved_r
        
        return preds

    # =========================================================================
    # Protected Optimization Hooks
    # =========================================================================

    def _update_state(self, key, S, target):
        Rk = self._Rk[key]
        self._phi[key] = self._gamma * self._phi[key] + (1 - self._gamma) * (S.T @ Rk @ S)
        self._r[key]   = self._gamma * self._r[key]   + (1 - self._gamma) * (S.T @ target)

    def _compute_step_size(self, key):
        if self._phi[key].size > 0:
            # Use eigvalsh for symmetric matrix stability
            maxeig = np.max(np.linalg.eigvalsh(self._phi[key]))
        else:
            maxeig = 0
        return 1.0 / (0.0001 + maxeig)

    def _apply_descent_step(self, key, eta):
        # self._grad = (self._phi[key] @ self._theta[key]) - self._r[key] + (self._lambda * self._theta[key])
        grad = self._get_gradient(key)
        self._theta[key] -= eta * grad

        if self._LassoEn:
            # Element-wise maximum for soft thresholding
            val = 1 - (eta * self._lambda) / (np.abs(self._theta[key]) + 1e-9)
            self._theta[key] *= np.maximum(0, val)

    def _get_gradient(self, key):
        return (self._phi[key] @ self._theta[key]) - self._r[key] + (self._lambda * self._theta[key])

    # =========================================================================
    # Generic Helpers
    # =========================================================================

    def _allocate_state_variables(self):
        for key in self._data_keys:
            feat_dim = self._features[key]["S_dim"]
            self._phi[key] = np.zeros((feat_dim, feat_dim))
            self._r[key] = np.zeros((feat_dim, 1))
            self._theta[key] = self._theta_initializer((feat_dim, 1))

    def __zero_initializer(self, theta_shape):
        return np.zeros(shape=theta_shape)

    def _data_initializer(self):
        self._data = dict()
        self._norm_scale = dict()
        self._bias = dict()
        for key in self._data_keys:
            self._data[key] = np.zeros(shape=(self._N[key], self._P))
            self._norm_scale[key] = 0
            if self._bias_enabler:
                self._bias[key] = np.ones(shape=(self._N[key], 1))
            else:
                self._bias[key] = np.empty(shape=(self._N[key], 0))

    def _feature_gen(self):
        """
        Generic Feature Generation.
        CRITICAL FIX: Normalization is Column-Wise (axis=0).
        """
        featureDict = dict()

        for key in self._data_keys:
            x_self = self._data[key]
            x_lower = self._data.get(key - 1) if (key - 1) in self._data else None
            x_upper = self._data.get(key + 1) if (key + 1) in self._data else None
            
            # Generate Raw Features
            S = self.__generic_features(key, x_self, x_lower, x_upper)

            # Normalization
            if self._FeatureNormalzn:
                # 1. Compute Squared Norm per Column (Vector of size FeatDim)
                # CRITICAL: axis=0 prevents scalar summation of the whole matrix
                S_n = np.sum(S**2, axis=0)
                
                # 2. Floor to prevent division by zero (Matches MATLAB 0.001)
                S_n[S_n == 0] = 0.001
                
                # 3. Compute Signal Variance (Scalar) of the most recent lag
                varV = np.sum(x_self[:, -1]**2)
                
                # 4. Update Running Scale
                self._norm_scale[key] = (1 - self._b) * self._norm_scale[key] + self._b * np.sqrt(varV)
                
                # 5. Broadcast Division and Scale
                # (N, F) / (F,) -> Each column divided by its norm
                featureDict[key] = (S / np.sqrt(S_n)) * self._norm_scale[key]
            else:
                featureDict[key] = S

        return featureDict

    def __generic_features(self, key, x_self, x_lower, x_upper):
        """
        Constructs S = [Neighbor_Lower, Self_Lower, Self_Upper, Neighbor_Upper, Bias]
        Order matched to MATLAB: [Lower, Self, Upper, Bias]
        """
        components = []
        
        # 1. Lower Neighbor Features (L * B_k^T * x_{k-1})
        if x_lower is not None and "l" in self._features[key]:
            components.append(self.__matrix_vector_bw(self._features[key]["l"], x_lower))

        # 2. Self Features (Lower Coupling)
        if "sl" in self._features[key]:
             components.append(self.__matrix_vector_bw(self._features[key]["sl"], x_self))
        
        # 3. Self Features (Upper Coupling)
        if "su" in self._features[key]:
             components.append(self.__matrix_vector_bw(self._features[key]["su"], x_self))
        
        # 4. Upper Neighbor Features (L * B_{k+1} * x_{k+1})
        if x_upper is not None and "u" in self._features[key]:
            components.append(self.__matrix_vector_bw(self._features[key]["u"], x_upper))
            
        # 5. Bias
        components.append(self._bias[key])
        
        return np.hstack(components)

    def _construct_laplacian(self, cellularComplex):
        self._N = dict()
        self._features = dict()
        self._Rk = dict()
        
        for k in self._data_keys:
            self._features[k] = {}
            feature_dim_accum = 0
            
            B_down = cellularComplex.get(k)
            B_up   = cellularComplex.get(k+1)
            
            if B_down is not None:
                self._N[k] = B_down.shape[1]
            elif B_up is not None:
                self._N[k] = B_up.shape[0]
            else:
                raise ValueError(f"Dim {k} enabled but no boundaries found.")

            L_lower = None
            L_upper = None
            
            if B_down is not None:
                L_lower = B_down.T @ B_down
            if B_up is not None:
                L_upper = B_up @ B_up.T
                
            mu_val = self._mu[k] if k < len(self._mu) else 0
            self._Rk[k] = np.eye(self._N[k])
            
            # --- Feature Dimension accumulation matches Order in __generic_features ---
            # 1. Lower Neighbor
            if L_lower is not None:
                K_val = self._K[k] if k < len(self._K) else 2
                K_l = K_val[0] if isinstance(K_val, (list, tuple)) else K_val
                
                self._features[k]["sl"] = self.__ll_gen(L_lower, K_l)
                
                # Check for Lower Neighbor existence
                if (k-1) in self._data_keys:
                    self._features[k]["l"] = self.__multiply_matrices_blockwise(self._features[k]["sl"], B_down.T)
                    feature_dim_accum += K_l * self._P
            
            # 2. Self (Lower)
            if L_lower is not None:
                 feature_dim_accum += K_l * self._P 

            # 3. Self (Upper)
            if L_upper is not None:
                K_val = self._K[k] if k < len(self._K) else 2
                K_u = K_val[1] if isinstance(K_val, (list, tuple)) else K_val
                
                self._features[k]["su"] = self.__ll_gen(L_upper, K_u)
                feature_dim_accum += K_u * self._P
            
            # 4. Upper Neighbor
            if L_upper is not None:
                if (k+1) in self._data_keys:
                    self._features[k]["u"] = self.__multiply_matrices_blockwise(self._features[k]["su"], B_up)
                    feature_dim_accum += K_u * self._P

            # 5. Bias
            feature_dim_accum += int(self._bias_enabler)
            
            # Add regularization for L_lower/L_upper
            if L_lower is not None:
                 mu_l = mu_val[0] if isinstance(mu_val, (list, tuple)) else mu_val
                 self._Rk[k] += mu_l * L_lower
            if L_upper is not None:
                 mu_u = mu_val[1] if isinstance(mu_val, (list, tuple)) else mu_val
                 self._Rk[k] += mu_u * L_upper
            
            self._features[k]["S_dim"] = feature_dim_accum

    def __ll_gen(self, L, K):
        LL = np.empty((L.shape[0], L.shape[1], K))
        curr_L = np.eye(L.shape[0])
        LL[:, :, 0] = curr_L
        for k in range(1, K):
            curr_L = curr_L @ L
            LL[:, :, k] = curr_L
        return LL

    @staticmethod
    def __multiply_matrices_blockwise(M_stack, B):
        K = M_stack.shape[2]
        MB_stack = np.empty((M_stack.shape[0], B.shape[1], K))
        for k in range(K):
            MB_stack[:, :, k] = M_stack[:, :, k] @ B
        return MB_stack

    @staticmethod
    def __matrix_vector_bw(MB_stack, X):

            # X shape: (N_in, P_lags) -> [Oldest ... Newest]
        
        # 1. Flip X along time axis (axis 1) to match MATLAB's loop order
        # New shape: [Newest ... Oldest]
        X_flipped = X[:, ::-1]
        
        # 2. Einsum
        # MB_stack: (N_out, N_in, K)
        # X_flipped: (N_in, P)
        # Output Y_out: (N_out, K, P)
        Y_out = np.einsum('nmk,mp->nkp', MB_stack, X_flipped)
        
        # 3. Flatten (Fortran order)
        # This keeps the first index (N) contiguous, then K, then P.
        # Since P is now [Newest...Oldest], the columns of S will be:
        # [K_block(Newest), K_block(2nd_Newest), ...]
        return np.reshape(Y_out, shape=(MB_stack.shape[0], -1), order='F')
        # Y_out = np.einsum('nmk,mp->nkp', MB_stack, X)
        # return np.reshape(Y_out, shape=(MB_stack.shape[0], -1), order='F')

    def __algorithm_parameter_setup(self, algorithmParam):
        self._Tstep = algorithmParam.get('Tstep', 1)
        self._mu = algorithmParam.get('mu', [0, (0,0), 0]) 
        self._lambda = algorithmParam.get('lambda', 0.01)
        self._LassoEn = algorithmParam.get('LassoEn', 0)
        self._FeatureNormalzn = algorithmParam.get('FeatureNormalzn', True)
        self._bias_enabler = algorithmParam.get('BiasEn', True)
        self._b = algorithmParam.get('b', 1)
        self._gamma = algorithmParam.get('gamma', 0.98)
        self._P = algorithmParam.get('P', 2)
        self._K = algorithmParam.get('K', [2, (2,2), 2]) 
        self._data_enabler = algorithmParam.get('enabler', [True, True, True])