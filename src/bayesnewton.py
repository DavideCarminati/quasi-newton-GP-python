# shebang

import SR1_solver
import system
import numpy as np
import numpy.typing as npt

class MarkovGP:
    def __init__(self,
                 observation: npt.NDArray,
                 A: npt.NDArray,
                 H: npt.NDArray,
                 P0: npt.NDArray,
                 Pinf: npt.NDArray,
                 Q: npt.NDArray) -> None:
        # !!!!!!!!!!! IL MATLAB ERA SBAGLIATO PERCHE' QUANDO OTTIMIZZO GLI IPERPARAMETRI NON AGGIORNO LE MATRICI A, Q E P MA SOLO LE LORO DERIVATE!!
        self.N = np.size(observation)
        self.y = np.array(observation)
        self.A = np.array(A)
        self.H = np.array(H)
        self.P0 = np.array(P0)
        self.Pinf = np.array(Pinf)
        self.Q = np.array(Q)
        self.m_x = np.zeros(self.N)
        # questi dopo mi sa...
        # self.m_x(:, 1) = sol
        # self.m_x(:, 2) = randn(TRAIN_POINTS, 1);
        # self.P_x = zeros(2, 2, TRAIN_POINTS);
        # self.P_x(:,:,1) = P0;
        # self.x_hat = zeros(TRAIN_POINTS, 2);
        # self.P_hat = zeros(2, 2, TRAIN_POINTS);

    def filter(self):
        pass

    def laplace_energy(self):
        pass

    def dlaplace_energy(self):
        pass