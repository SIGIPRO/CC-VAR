import numpy as np


class CCVAR:

    def __init__(self, algorithmParam, cellularComplex):
        self.__algorithm_parameter_setup(algorithmParam)
        
        self._data_keys = [i for i in range(len(self._data_enabler)) if self._data_enabler[i]]
     
        self.__construct_laplacian(cellularComplex)
      
        # self._cc = cellularComplex

        # self._theta = np.zeros(shape = (self._paramNum,))

        # self._data_keys = [i in range]
        # for i in range(len(self._data_enabler)):
        #     self._data_keys.append()
        # self._data_keys = []
        # if self._V_En:
        #     self._data_keys.append("l")
        # if self._F_En:
        #     self._data_keys.append("s")
        # if self._T_En:
        #     self._data_keys.append("u")

        self._data_initializer()

        self._phi = dict()
        self._r = dict()
        self._theta = dict()
        for key in self._data_keys:
            self._phi[key] = 0
            self._r[key] = 0
            self._theta[key] = 0
            
        

        # assert self.__check_initial(initializer)

    def _data_initializer(self):
        ## NOTE Inherit the class to change the initializer
        self._data = dict()

        for key in self._data_keys:
            self._data[key] =  np.zeros(shape = (self._N[key], self._P))
        # if "l" in self._data_keys:
        #     self._data["l"] = np.zeros(shape = (self._Hodge["N0"], self._P))
        # if "s" in self._data_keys:
        #     self._data["s"] = np.zeros(shape = (self._Hodge["N1"], self._P))
        # if "u"
        # self._data["u"] = np.zeros(shape = (self._Hodge["N2"], self._P))

    @classmethod
    def fromDataName(cls, dataName):

        return NotImplementedError
    
    ## TODO Think about the prediction function again.
    # def updateAndPredict(self, inputData):
    #     self.update(inputData)
    #     self.predictData(inputData)
    # def predictData(self):
    #     pass
    def update(self, inputData):

        # estimatorParam{i}.maxeig=max(eigs(estimatorParam{i}.PHI));
        self.__phi_r_gen(inputData=inputData)

        for key in self._data_keys:
            maxeig = np.max(np.linalg.eigvalsh(self._phi[key]))
            eta=2/(0.0001 + maxeig)
            self._theta[key] -= eta*(self._phi[key] @ self._theta[key] - self._r[key] + self._lambda * self._theta[key])
            self._data[key] = np.roll(self._data[key], shift = 1, axis = 1)
            self._data[key][:,0] = inputData[key]

    def predict(self):

        featureDict = self.__feature_gen()
        predictions = dict()

        for key in self._data_keys:
            theta = np.asarray(self._theta[key])
            if theta.ndim == 0:
                predictions[key] = np.zeros(self._N[key])
                continue
            if theta.ndim > 1 and theta.shape[1] == 1:
                theta = theta.reshape(-1)
            predictions[key] = featureDict[key] @ theta

        return predictions
    
    def __algorithm_parameter_setup(self, algorithmParam):
  
        self._Tstep = algorithmParam.get('Tstep', 1)
        # self._mu_0 = algorithmParam.get('mu_0', 0)
        # self._mu_1l = algorithmParam.get('mu_1l', 0)
        # self._mu_1u = algorithmParam.get('mu_1u', 0)
        # self._mu_2 = algorithmParam.get('mu_2', 0)
        self._mu = algorithmParam.get('mu', [(0), (0,0), (0)])
        self._lambda = algorithmParam.get('lambda', 0.01)
        self._LassoEn = algorithmParam.get('LassoEn', 0)
        self._HodgeNormalzn = algorithmParam.get('HodgeNormalzn', False)
        self._FeatureNormalzn = algorithmParam.get('FeatureNormalzn', True)
        self._b = algorithmParam.get('b', 1)
        self._gamma = algorithmParam.get('gamma', 0.98)
        self._delta = 1 - self._gamma

        self._P = algorithmParam.get('P', 2)
        self._K = algorithmParam.get('K', ((2), (2,2,4), (2)))
        # self._K = algorithmParam.get('K', {'K0' : 2, 'K1': 4, 'K1l': 2, 'K1u': 2, 'K2' : 2})

        self._data_enabler = algorithmParam.get('enabler', [True, True, True])

        # self._V_En = algorithmParam.get('V_En', True)
        # self._F_En = algorithmParam.get('F_En', True)
        # self._T_En = algorithmParam.get('T_En', True)

        
        

    def __check_cc(self, cellularComplex):
        for key in self._data_keys:
            assert key in cellularComplex
        # assert "B1" in cellularComplex
        # assert "B2" in cellularComplex

    def __ll_gen(self, L, K):

        assert K!=0, "K should be positive!!!"

        LL = np.hstack([np.linalg.matrix_power(L, k) for k in range(K)])
            
        if self._HodgeNormalzn:
            U, S, Vh = np.linalg.svd(LL, full_matrices=False)
            S = S/np.max(S)
            LL = np.dot(U * S, Vh)

        
        return np.reshape(LL, shape=(L.shape[0], L.shape[0], K), order = 'F')
    
    def __phi_r_gen(self, inputData):
        featureDict = self.__feature_gen()
        
        for key in self._data_keys:
            currFeature = featureDict[key]
            Rk = self._Rk[key].copy()
            Rk.diagonal()[:]+=1
            self._phi[key] = self._gamma * self._phi[key] + (1 - self._gamma) * currFeature.T @ (Rk) @ currFeature

            self._r[key] = self._gamma * self._r[key] + (1-self._gamma) * currFeature.T @ inputData[key][:,-1]
        
    
    def __feature_gen(self):
    
        # x_0 = inputData["l"][:, 0:self._P]
        # x_1 = inputData["s"][:, 0:self._P]
        # x_2 = inputData["u"][:, 0:self._P]

        # featureDict = {"l":  self.__vertex_features(x_0, x_1),
        #                "s": self.__edge_features(x_0,x_1,x_2),
        #                "u": self.__polygon_features(x_1,x_2)}

        featureDict = dict()

        for key in self._data_keys:
            if key == 0:
                x_0 = self._data[0]
                x_1 = self._data.get(1)
                featureDict[key] = self.__vertex_features(x_0, x_1)

            elif key == self._cc_dim:
                x_1 = self._data.get(key - 1)
                x_2 = self._data[key]
                featureDict[key] = self.__polygon_features(x_1, x_2)
            else:
                x_0 = self._data.get(key - 1)
                x_1 = self._data[key]
                x_2 = self._data.get(key + 1)
                featureDict[key] = self.__edge_features(x_0, x_1, x_2, key)

        return featureDict

    def __vertex_features(self, x_0, x_1):
        vfself = self.__matrix_vector_bw(self._features[0]["s"], x_0)
        vfupper = np.array([])

        if x_1 is not None:
            vfupper = self.__matrix_vector_bw(self._features[0]["u"], x_1)
            
        return np.hstack([vfself, vfupper])
    

       
    
    def __edge_features(self, x_0, x_1, x_2, key):
        eflower = np.array([])
        efupper = np.array([])

        efselflower =  self.__matrix_vector_bw(self._features[key]["sl"], x_1)
        efselfupper = self.__matrix_vector_bw(self._features[key]["su"], x_1)

        if x_0 is not None:
            eflower = self.__matrix_vector_bw(self._features[key]["l"], x_0)

        if x_2 is not None:
            efupper = self.__matrix_vector_bw(self._features[key]["u"], x_2)

        return np.hstack([efselflower, efselfupper, eflower, efupper])
    

    def __polygon_features(self, x_1, x_2):
        tfself = self.__matrix_vector_bw(self._features[self._cc_dim]["s"], x_2)
        tflower = np.array([])

        if x_1 is not None:
            tflower = self.__matrix_vector_bw(self._features[self._cc_dim - 1], x_1)
            
        return np.hstack([tfself, tflower])
            
        
       
    @staticmethod
    def __matrix_vector_bw(MB_stack, X):
        Y_out = np.einsum('nmk,mp->nkp', MB_stack, X)
        return np.reshape(Y_out, shape = (MB_stack.shape[0], -1), order = 'F')

    @staticmethod
    def __multiply_matrices_blockwise(M, B, nB):
        MB_stack = np.empty((M.shape[0], B.shape[1], nB))
        for b in nB:
            MB_stack[:, :, b] = M[:, :, b] @ B
        
        return MB_stack


    def __construct_laplacian(self, cellularComplex):
        self.__check_cc(cellularComplex)
        # B1 = cellularComplex["B1"]
        # B2 = cellularComplex["B2"]
        # self._Hodge = dict()
        # self._Hodge["boundary"] = dict()
        # self._Hodge["laplacian"] = dict()
        # self._Hodge["laplacian_powers"] = dict()
        # self._Hodge["N"] = dict()
        # self._Hodge["features"] = dict()
        self._N = dict()
        self._features = dict()
        self._Rk = dict()
        self._cc_dim = max(self._data_keys)

        def lower_laplacian(boundary):
            return boundary.T @ boundary
        
        def upper_laplacian(boundary):
            return boundary @ boundary.T

        for key in self._data_keys:
            # self._Hodge["boundary"][key] = cellularComplex[key]
            self._features[key] = dict()

            
            if key == 0:
                # self._Hodge["laplacian"][key] = (upper_laplacian(cellularComplex[key]))
                boundary_1 = cellularComplex[key]
                up_lap = upper_laplacian(boundary_1)
        
                self._Rk[key] = self._mu[key][0] * up_lap
                self._features[key]["s"] = self.__ll_gen(up_lap, self._K[key][0])
                self._features[key]["u"] = self.__multiply_matrices_blockwise(self._features[key]["s"], boundary_1, self._K[key][0])

                self._N[key] = up_lap.shape[0]
                
               
            elif key == self._cc_dim:
                # up_lap = upper_laplacian(cellularComplex[key])
                
                boundary_m1 = cellularComplex[key - 1]
                low_lap = lower_laplacian(boundary_m1)
                # self._Hodge["laplacian"][key] = (lower_laplacian(cellularComplex[key-1]))
                self._Rk[key] = self._mu[key][0] * low_lap
                self._features[key]["s"] = self.__ll_gen(low_lap, self._K[key][0])
                self._features["l"] = self.__multiply_matrices_blockwise(self._features[key]["s"], boundary_m1.T, self._K[key][0])

                self._N[key] = low_lap.shape[0]
            else:
                boundary_1 = cellularComplex[key]
                up_lap = upper_laplacian(boundary_1)
                boundary_m1 = cellularComplex[key - 1]
                low_lap = lower_laplacian(boundary_m1)


                # self._Hodge["laplacian"][key] = (lower_laplacian(cellularComplex[key - 1]), upper_laplacian(cellularComplex[key]))
                self._Rk[key] = self._mu[key][0] * low_lap + self._mu[key][1] * up_lap
                self._features["sl"] =  self.__ll_gen(low_lap, self._K[key][0])
                self._features["su"] =  self.__ll_gen(up_lap, self._K[key][1])
                self._features["l"] = self.__multiply_matrices_blockwise(self._features[key]["sl"], boundary_m1.T, self._K[key][0])
                self._features["u"] = self.__multiply_matrices_blockwise(self._features[key]["su"], boundary_1, self._K[key][1])

                self._N[key] = low_lap.shape[0]


            # self._N[key] = self._Hodge["laplacian"][key][0].shape[0]
            # self._N[key] = 
            # self._Hodge["laplacian_powers"][key] = [self.__ll_gen(self._Hodge["laplacian"][i], self._K[key][i]) for i in range(len(self._Hodge["laplacian"]))]
            # self._Rk[key] = 0
            # self._Hodge["laplacian_powers"][key] = []
            # for i in range(len(self._Hodge["laplacian"])):
            #     # self._Rk[key] += self._mu[key][i] * self._Hodge["laplacian"][key][i]
            #     self._Hodge["laplacian_powers"][key].append(self.__ll_gen(self._Hodge["laplacian"][i], self._K[key][i]))

            

        # self._Hodge["B1"] = B1
        # self._Hodge["B2"] = B2
        # self._Hodge["L0"] = B1 @ B1.T
        # self._Hodge["L1_lwr"] = B1.T @ B1
        # self._Hodge["L1_upr"] = B2 @ B2.T
        # self._Hodge["L2"] = B2.T @ B2
        # self._Hodge["N0"] = np.shape(B1)[0]
        # self._Hodge["N1"] = np.shape(B1)[1]
        # self._Hodge["N2"] = np.shape(B2)[1]

        # self._Hodge["LL0"] = self.__ll_gen(self._Hodge["L0"], self._K["K0"])
        # self._Hodge["LL1_lwr"] = self.__ll_gen(self._Hodge["L1_lwr"], self._K["K1l"])
        # self._Hodge["LL1_upr"] = self.__ll_gen(self._Hodge["L1_upr"], self._K["K1u"])
        # self._Hodge["LL2"] = self.__ll_gen(self._Hodge["L2"], self._K["K2"])


        # if self._V_En:
        #     self._Hodge["vself"] = self._Hodge["LL0"]
        #     self._Hodge["vupper"] = self.__multiply_matrices_blockwise(self._Hodge["LL0"], self._Hodge["B1"], self._Hodge["K0"])

        # if self._F_En:
        #     self._Hodge["elower"] = self.__multiply_matrices_blockwise(self._Hodge["LL1_lwr"], self._Hodge["B1"].T, self._Hodge["K1l"])
        #     self._Hodge["eupper"] = self.__multiply_matrices_blockwise(self._Hodge["LL1_upr"], self._Hodge["B2"], self._Hodge["K1u"])
        #     self._Hodge["eselflower"] = self._Hodge["LL1_lwr"]
        #     self._Hodge["eselfupper"] = self._Hodge["LL1_upr"]


        # if self._T_En:
        #     self._Hodge["tself"] = self._Hodge["LL2"]
        #     self._Hodge["tlower"] = self.__multiply_matrices_blockwise(self._Hodge["LL2"], self._Hodge["B2"].T, self._Hodge["K2"])
        # self._Rk = {"l":self._mu_0 * self._Hodge["L0"],
        #             "s":self._mu_1l * self._Hodge["L1_lwr"] + self._mu_1u * self._Hodge["L1_upr"], 
        #             "u":self._mu_2 * self._Hodge["L2"]}
        # self._cc_dim = 2
