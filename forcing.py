

import numpy as np

def incomp_shell_2d(kf, kfw, dkx, dky, seed=None):
    # Random number generator
    rand = np.random.RandomState(seed)
    # Return draw function
    def draw(kx, ky, kf=kf, kfw=kfw, dkx=dkx, dky=dky, rand=rand):
        k = (kx**2 + ky**2)**0.5
        # 1D power spectrum: normalized Gaussian, no mean
        E_k1 = np.exp(-(k-kf)**2/2/kfw**2) / kfw / np.sqrt(2*np.pi) * (k != 0)
        # 2D power spectrum: divide by polar Jacobian
        E_k2 = E_k1 / np.pi / (k + (k==0))
        # Forcing amplitude with random phase
        phase = 2 * np.pi * rand.rand(*k.shape)
        f_amp = (E_k2 * dkx * dky)**0.5 * np.exp(1j*phase)
        f_amp *= rand.randn(*k.shape)
        # Forcing components: divergence free
        fx = ky * f_amp / (k + (k==0))
        fy = - kx * f_amp / (k + (k==0))
        return fx, fy
    return draw

