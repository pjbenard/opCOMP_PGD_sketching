import numpy as np

class Gaussian:
    def __init__(self, **parameters):
        super().__init__()

        default_params = {"m": 200, "d": 1, "lamb": 0.1}
        for key, value in default_params.items():
            setattr(self, key, value)

        for key, value in parameters.items():
            setattr(self, key, value)

        if "bounds" not in parameters.keys():
            self.bounds = {
                "min": np.array((0,) * self.d),
                "max": np.array((1,) * self.d),
            }

        self.w = np.random.randn(self.m, self.d) / self.lamb

    def Adelta(self, t):
        """
        (k, m)
        """
        expval = np.exp(-1j * np.dot(t, self.w.T))

        return expval / self.m**0.5

    def Adeltap(self, t):
        """
        (d, k, m)
        """
        dA_dt = 1j * self.w.T[:, None, :] * self.Adelta(t)[None, :, :]

        return dA_dt

    def Adeltapp(self, t):
        """
        (d, d, k, m)
        """
        ddA_ddt = (
            -self.w.T[None, :, None, :]
            * self.w.T[:, None, None, :]
            * self.Adelta(t)[None, None, :, :]
        )

        return ddA_ddt

    def Ax(self, a, t):
        """
        (m)
        """
        expval = self.Adelta(t)

        return np.dot(a, expval)

    def image(self, grid, y=None, a=None, t=None):
        if y is None:
            y = self.Ax(a, t)

        m = np.size(y)
        dims_size = np.shape(grid)[:-1]

        expval = 1 / self.Adelta(grid.reshape((-1, self.d)))

        z = np.dot(expval, y).reshape(dims_size)

        return np.abs(z) / m  # Modulus of complex numbers

    def hermitian_prod(self, u, v):
        return np.sum(u * np.conj(v))