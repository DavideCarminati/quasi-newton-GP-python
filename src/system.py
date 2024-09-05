import numpy as np
import numpy.typing as npt
import scipy
from scipy import signal
import scipy.linalg
import scipy.signal

# I need scipy signals for lsim method!!!

class System:
    def __init__(self) -> None:
        
        m = 1
        k = 1
        b = 0.2
        self.A = np.array([[0, 1], [-k/m, -b/m]])
        self.B = np.array([[0, 1/m]]).T
        self.C = np.array([[1, 0]])
        self.D = np.array([0])

        # self.st_dev_noise = np.sqrt(1e-3)
        # self.lengthscale = 1.5

        self.dynamical_sys = scipy.signal.lti(self.A, self.B, self.C, self.D)

    def simulate(self, time_steps=1000, t_end=15, st_dev_noise=0.0) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

        self.time_steps = time_steps
        time = np.linspace(0, t_end, self.time_steps)
        u_vet = np.zeros(self.time_steps)
        # Integration
        [self.times, self.y_noiseless, self.x] = scipy.signal.lsim(self.dynamical_sys, u_vet, time, X0=[1, 0])

        # Add sensor noise
        self.y = self.y_noiseless + st_dev_noise * np.random.randn(self.time_steps)

        return self.y, self.y_noiseless, self.times, self.x
    
    def matern32_to_lti(self, hyperparameters: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        # Define SDE matrices using Matern 3/2 kernel

        self.lengthscale = hyperparameters[0]
        self.variance = hyperparameters[1]**2
        
        lam = 3.0**0.5 / self.lengthscale
        F = np.array([[0.0, 1.0],
            [-lam**2, -2 * lam]])
        L = np.array([0, 1])
        # Q = [12.0 * 3.0 ^ 0.5 / lengthscale ^ 3.0 * variance]; % Actually this is spectral density of beta. We need Qn
        H = np.array([1.0, 0.0])
        Pinf = np.array([[self.variance, 0.0],
                [0.0, 3.0 * self.variance / self.lengthscale**2.0]])
        P0 = Pinf
        # state_transition(self, dt):
            
        #     Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        #     :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        #     :return: state transition matrix A [2, 2]

        dt = self.times[1] - self.times[0]
        A_tmp = np.array([[lam, 1.0], [-lam**2.0, -lam]])
        A = np.array(scipy.linalg.expm(-dt * lam) * (dt * A_tmp + np.identity(2)))

        Q = Pinf - A @ Pinf @ A.T

        return A, H, P0, Pinf, Q