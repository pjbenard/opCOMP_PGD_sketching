import numpy as np

class Gaussian2D_MATIRF:
    def __init__GRID(self):
        half_len_pixX = (self.b1 / self.N1) / 2
        half_len_pixY = (self.b2 / self.N2) / 2
        X_range = np.linspace(half_len_pixX, self.b1 - half_len_pixX, num=self.N1)
        Y_range = np.linspace(half_len_pixY, self.b2 - half_len_pixY, num=self.N2)
        grid = np.meshgrid(X_range, Y_range)
        self.grid = np.stack(grid, axis=-1)
        self.grid_flat = np.reshape(self.grid, (-1, 2))
    
    def __init__(self, sigma_x=None, sigma_y=None, **parameters):

        default_params = {
            "K": 4,
            "b1": 6.4,
            "b2": 6.4,
            "b3": 0.8,
            "d": 3,
            "N1": 64,
            "N2": 64,
            "NA": 1.49,
            "ni": 1.515,
            "nt": 1.333,
            "lambda_l": 0.66,
        }
        for key, value in default_params.items():
            setattr(self, key, value)

        for key, value in parameters.items():
            setattr(self, key, value)

        if sigma_x is None:
            self.sigma_x = 0.42 * self.lambda_l / self.NA
        else:
            self.sigma_x = sigma_x

        if sigma_y is None:
            self.sigma_y = 0.42 * self.lambda_l / self.NA
        else:
            self.sigma_y = sigma_y

        self.__init__GRID()

        # self.sigma_x = self.sigma_y = 0.42 * self.lambda_l / self.NA
        self.sigma = np.array([self.sigma_x, self.sigma_y])
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
    
    def _incidence_coef(self, z):
        """
        (k, K)
        """
        cst = 2 * np.pi * self.sigma_x * self.sigma_y
        return self._xi(z)[:, None] * np.exp(- self.s_k[None, :] * z[:, None]) / cst
    
    def _gaussian_2D(self, t):
        """
        (k, N1*N2)
        """
        subts = (self.grid_flat[None, ...] - t[:, None, :]) / (2**.5 * self.sigma)
        expval = np.exp(- np.einsum('ijk, ijk->ij', subts, subts))
        return expval
    
    def Adelta(self, t, reshape=True):
        """
        reshape == True  : (k, m=N1*N2*K)
        reshape == False : (k, N1*N2, K)
        """
        # expval = self._gaussian_2D(t[:, :2])[:, :, None]
        # incoef = self._incidence_coef(t[:, -1])[:, None, :]
        # out = expval * incoef
        expval = self._gaussian_2D(t[:, :2])
        incoef = self._incidence_coef(t[:, -1])
        out = np.einsum('ij,ik->ijk', expval, incoef)

        if reshape:
            return out.reshape((out.shape[0], -1))

        return out

    def Adeltap(self, t):
        """
        (d, k, N1*N2*K)
        """
        phi = self.Adelta(t, reshape=False) # (k, N1*N2, K)
        dA_dt = np.zeros((self.d, *phi.shape))

        subts = self.grid_flat[:, None, :] - t[None, :, :2] # (N1*N2, k, d)
        dA_dt[:2] = (subts.T / self.sigma[:, None, None]**2)[..., None] * phi[None, ...] # (d, k, N1*N2, K)

        coef_dA_dz = (self._xi(t[:, -1])**2 * np.dot(np.exp(- 2 * self.s_k[None, :] * t[:, -1][:, None]), self.s_k))[:, None] - self.s_k[None, :] # (k, K)
        dA_dt[-1] = coef_dA_dz[:, None, :] * phi

        dA_dt_reshaped = np.reshape(dA_dt, (*dA_dt.shape[:2], -1))
        return -dA_dt_reshaped
    
    def Ax(self, a, t):
        """
        (m)
        """
        expval = self.Adelta(t)
        y_flat = np.dot(a, expval)
        return y_flat

    def image(self, y):
        return np.reshape(y, (*self.grid.shape[:-1], self.K))