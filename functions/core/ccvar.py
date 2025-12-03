import numpy as np
import copy

class CCVAR:
    def __init__(self, algorithmParam, cellularComplex, theta_initializer=None):
        self.__algorithm_parameter_setup(algorithmParam)
        
        # Generic keys: [0, 1, 2, 3...]
        self._data_keys = sorted([i for i, en in enumerate(self._data_enabler) if en])
        
        # Initialize storage
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
        # Default: Cold start (reset everything)
        self._allocate_state_variables()
        self._data_initializer()

    def update(self, inputData):
        """Online Learning Loop."""
        featureDict = self._feature_gen()
        
        for key in self._data_keys:
            if key not in inputData: continue
            
            S = featureDict[key]
            target = inputData[key].reshape(-1, 1)
            
            # --- Hooks for Distributed/Gradient Logic ---
            self._update_state(key, S, target)
            eta = self._compute_step_size(key)
            self._apply_descent_step(key, eta)

            # Update History (Slide Window)
            self._data[key] = np.roll(self._data[key], shift=-1, axis=1)
            self._data[key][:, -1] = inputData[key].flatten()

    def forecast(self, steps):
        """Recursive N-step forecasting."""
        preds = {k: np.zeros((self._N[k], steps)) for k in self._data_keys}
        
        saved_data = copy.deepcopy(self._data)
        saved_norm = copy.deepcopy(self._norm_scale)
        
        # Reset norm for prediction
        for k in self._data_keys:
            self._norm_scale[k] = 0

        for t in range(steps):
            feats = self._feature_gen()
            current_step_pred = {}
            
            for k in self._data_keys:
                if self._theta[k] is None: continue
                y_pred = feats[k] @ self._theta[k]
                preds[k][:, t] = y_pred.flatten()
                current_step_pred[k] = y_pred.flatten()
            
            if t < steps - 1:
                for k in self._data_keys:
                    self._data[k] = np.roll(self._data[k], shift=-1, axis=1)
                    self._data[k][:, -1] = current_step_pred[k]

        self._data = saved_data
        self._norm_scale = saved_norm
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
            maxeig = np.max(np.linalg.eigvalsh(self._phi[key]))
        else:
            maxeig = 0
        return 1.0 / (0.0001 + maxeig)

    def _apply_descent_step(self, key, eta):
        grad = (self._phi[key] @ self._theta[key]) - self._r[key] + (self._lambda * self._theta[key])
        self._theta[key] -= eta * grad

        if self._LassoEn:
            val = 1 - (eta * self._lambda) / (np.abs(self._theta[key]) + 1e-9)
            self._theta[key] *= np.maximum(0, val)

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
        Generic Feature Generation for N-dimensions.
        Automatically finds Lower (k-1) and Upper (k+1) neighbors if they exist.
        """
        featureDict = dict()

        for key in self._data_keys:
            x_self = self._data[key]
            
            # Dynamic Neighbor Retrieval
            # If key-1 exists in our data, get it.
            x_lower = self._data.get(key - 1) if (key - 1) in self._data else None
            
            # If key+1 exists in our data, get it.
            x_upper = self._data.get(key + 1) if (key + 1) in self._data else None
            
            # Generate features generically
            featureDict[key] = self.__generic_features(key, x_self, x_lower, x_upper)

            # Normalization
            if self._FeatureNormalzn:
                S = featureDict[key]
                S_n = np.sum(S**2)
                if S_n == 0: S_n = 0.001
                varV = np.sum(x_self[:, -1]**2)
                self._norm_scale[key] = (1 - self._b) * self._norm_scale[key] + self._b * np.sqrt(varV)
                featureDict[key] = (S / np.sqrt(S_n)) * self._norm_scale[key]

        return featureDict

    def __generic_features(self, key, x_self, x_lower, x_upper):
        """
        Constructs [Self_Lower, Self_Upper, Neighbor_Lower, Neighbor_Upper, Bias]
        """
        components = []
        
        # 1. Self Features (Lower Coupling)
        if "sl" in self._features[key]:
             components.append(self.__matrix_vector_bw(self._features[key]["sl"], x_self))
        
        # 2. Self Features (Upper Coupling)
        if "su" in self._features[key]:
             components.append(self.__matrix_vector_bw(self._features[key]["su"], x_self))
        
        # 3. Lower Neighbor Features (L * B_k^T * x_{k-1})
        if x_lower is not None and "l" in self._features[key]:
            components.append(self.__matrix_vector_bw(self._features[key]["l"], x_lower))

        # 4. Upper Neighbor Features (L * B_{k+1} * x_{k+1})
        if x_upper is not None and "u" in self._features[key]:
            components.append(self.__matrix_vector_bw(self._features[key]["u"], x_upper))
            
        # 5. Bias
        components.append(self._bias[key])
        
        return np.hstack(components)

    def _construct_laplacian(self, cellularComplex):
        """
        Generic N-Dimensional Topology Construction.
        For any dimension 'k', looks for boundaries at 'k' and 'k+1'.
        """
        self._N = dict()
        self._features = dict()
        self._Rk = dict()
        
        for k in self._data_keys:
            self._features[k] = {}
            feature_dim_accum = 0
            
            # --- Identify Boundaries ---
            # B_down connects k -> k-1 (Stored at cellularComplex[k])
            # B_up   connects k+1 -> k (Stored at cellularComplex[k+1])
            B_down = cellularComplex.get(k)
            B_up   = cellularComplex.get(k+1)
            
            # We determine N from the available boundaries
            if B_down is not None:
                self._N[k] = B_down.shape[1]  # Cols of B_k are k-simplices
            elif B_up is not None:
                self._N[k] = B_up.shape[0]    # Rows of B_{k+1} are k-simplices
            else:
                raise ValueError(f"Dimension {k} enabled but no connecting boundaries found in complex.")

            # --- Compute Laplacians ---
            L_lower = None
            L_upper = None
            
            if B_down is not None:
                L_lower = B_down.T @ B_down
                
            if B_up is not None:
                L_upper = B_up @ B_up.T
                
            # --- Regularization Matrix Rk ---
            # Mu logic: if tuple, [0]=lower, [1]=upper. If scalar, applied to whatever exists.
            mu_val = self._mu[k] if k < len(self._mu) else 0
            
            self._Rk[k] = np.eye(self._N[k])
            
            if L_lower is not None:
                mu_l = mu_val[0] if isinstance(mu_val, (list, tuple)) else mu_val
                self._Rk[k] += mu_l * L_lower
                # Generate Features
                # Note: K might be tuple (K_lower, K_upper) or scalar
                K_val = self._K[k] if k < len(self._K) else 2
                K_l = K_val[0] if isinstance(K_val, (list, tuple)) else K_val
                
                self._features[k]["sl"] = self.__ll_gen(L_lower, K_l)
                # Coupling from Lower Neighbor (k-1)
                if (k-1) in self._data_keys:
                    self._features[k]["l"] = self.__multiply_matrices_blockwise(self._features[k]["sl"], B_down.T)
                    feature_dim_accum += K_l * self._P
                
                feature_dim_accum += K_l * self._P # For Self-Lower

            if L_upper is not None:
                mu_u = mu_val[1] if isinstance(mu_val, (list, tuple)) else mu_val
                self._Rk[k] += mu_u * L_upper
                
                K_val = self._K[k] if k < len(self._K) else 2
                K_u = K_val[1] if isinstance(K_val, (list, tuple)) else K_val
                
                self._features[k]["su"] = self.__ll_gen(L_upper, K_u)
                # Coupling from Upper Neighbor (k+1)
                if (k+1) in self._data_keys:
                    self._features[k]["u"] = self.__multiply_matrices_blockwise(self._features[k]["su"], B_up)
                    feature_dim_accum += K_u * self._P

                feature_dim_accum += K_u * self._P # For Self-Upper
            
            # --- Special Case for Top/Bottom specific naming compatibility ---
            # If only one Laplacian exists, your original code might have mapped it to "s" (self).
            # To keep generic logic simple, we use "sl" and "su". 
            # If strictly L_upper exists (Node), we rely on "su".
            # If strictly L_lower exists (Top Poly), we rely on "sl".
            
            feature_dim_accum += int(self._bias_enabler)
            self._features[k]["S_dim"] = feature_dim_accum

    # --- Helpers (Unchanged) ---
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
        Y_out = np.einsum('nmk,mp->nkp', MB_stack, X)
        return np.reshape(Y_out, shape=(MB_stack.shape[0], -1), order='F')

    def __algorithm_parameter_setup(self, algorithmParam):
        # Unchanged from previous version
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