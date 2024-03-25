import numpy as np

class Sketching_Gaussian2D_MATIRF:
    def __init__(self, m, sigma_x, sigma_y, **parameters):
        default_params = {
            "K": 4,
            "b1": 6.4,
            "b2": 6.4,
            "b3": 0.8,
            "d": 3,
            "NA": 1.49,
            "ni": 1.515,
            "nt": 1.333,
            "lambda_l": 0.66,
        }
        for key, value in default_params.items():
            setattr(self, key, value)

        for key, value in parameters.items():
            setattr(self, key, value)

        self.m = m
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma = np.array([self.sigma_x, self.sigma_y])
        self.w = np.random.randn(self.m, self.d - 1) / self.sigma[None, :]

        # self.sigma_x = self.sigma_y = 0.42 * self.lambda_l / self.NA
        self.alpha_max = np.arcsin(self.NA / self.ni)
        self.alpha_crit = np.arcsin(self.nt / self.ni)
        self.alpha_k = self.alpha_crit + np.arange(0, self.K) * (
            self.alpha_max - self.alpha_crit
        ) / (self.K - 1)

        self.s_k = (
            (np.sin(self.alpha_k) ** 2 - np.sin(self.alpha_crit) ** 2)
            * (4 * np.pi * self.ni)
            / self.lambda_l
        )

        if "bounds" not in parameters.keys():
            self.bounds = {
                "min": np.array((0,) * self.d),
                "max": np.array((self.b1, self.b2, self.b3)),
            }

    def _xi(self, z):
        """
        (k)
        """
        return np.power(np.sum(np.exp(- 2 * self.s_k[None, :] * z[:, None]), axis=-1), -1/2)
    
    def Adelta(self, t, reshape=True):
        """
        (k, m*K)
        """
        cst_xi = self._xi(t[:, -1]) # (k)
        cst_exp_sk = np.exp(-self.s_k[None, :] * t[:, -1][:, None]) # (k, K)
        cst_exp_om = np.exp(- np.dot(self.w**2, self.sigma**2) / 2) # (m)
        expval = np.exp(-1j * np.dot(t[:, :-1], self.w.T)) # (k, m)
        out = cst_xi[:, None, None] * cst_exp_sk[:, None, :] * cst_exp_om[None, :, None] * expval[:, :, None]
        if reshape:
            out_reshaped = np.reshape(out, (t.shape[0], -1))
            return out_reshaped

        return out

    def Adeltap(self, t):
        phi = self.Adelta(t, reshape=False)
        dA_dt = np.zeros((self.d, t.shape[0], self.m, self.K), dtype=complex)

        dA_dt[:2] = -1j * self.w.T[:, None, :, None] * phi[None, ...]

        coef_dA_dz = (self._xi(t[:, -1])**2 * np.dot(np.exp(- 2 * self.s_k[None, :] * t[:, -1][:, None]), self.s_k))[:, None] - self.s_k[None, :] # (k, K)
        dA_dt[-1] = coef_dA_dz[:, None, :] * phi
        
        dA_dt_reshaped = np.reshape(dA_dt, (*dA_dt.shape[:2], -1))
        return -dA_dt_reshaped / self.m

    def Ax(self, a, t):
        """
        (m)
        """
        expval = self.Adelta(t)
        return np.dot(a, expval)      

    def DFT_DiracComb(self, a_grid, t_grid):
        a_grid_flat = np.reshape(a_grid, (-1, self.K)) # (N1*N2, K)
        t_grid_flat = np.reshape(t_grid, (-1, self.d - 1)) # (k, d)
        T_inter = (t_grid[-1, -1] - t_grid[0, 0]) / (np.array(t_grid.shape[:-1]) - 1)
        expval = np.exp(-1j * np.dot(t_grid_flat, self.w.T)) * np.prod(T_inter) # (k, m)
        y_resh = np.dot(expval.T, a_grid_flat)
        y = np.ravel(y_resh)
        return y

    def image(self, grid, y=None, a=None, t=None):
        if y is None:
            y = self.Ax(a, t)

        grid_resh = np.reshape(grid, (-1, grid.shape[-1])) # (k, m)
        y_resh = np.reshape(y, (self.m, self.K)) # (m, K)

        cst_sigma = 1 / (2 * np.pi * self.sigma_x * self.sigma_y)
        cst_exp_om = np.exp(np.dot(self.w**2, self.sigma**2) / 2) # (m)
        expval = cst_sigma * cst_exp_om[None, :] * np.exp(1j * np.dot(grid_resh, self.w.T)) # (k, m)
        # expval = 1 / self.Adelta(grid.reshape((-1, self.d)))

        z = np.dot(expval, y_resh).reshape(*grid.shape[:-1], self.K)
        
        return np.abs(z) / self.m  # Modulus of complex numbers