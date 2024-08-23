import numpy as np
from numpy.linalg import inv
import numpy.typing as npt
import scipy
import scipy.stats
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go

import utils
from SR1_solver import SR1_solver


class MarkovGP:
    def __init__(self,
                 time: npt.NDArray,
                 observation: npt.NDArray,
                 A: npt.NDArray,
                 H: npt.NDArray,
                 P0: npt.NDArray,
                 Pinf: npt.NDArray,
                 Q: npt.NDArray,
                 hyperparameters: npt.NDArray) -> None:
        # !!!!!!!!!!! IL MATLAB ERA SBAGLIATO PERCHE' QUANDO OTTIMIZZO GLI IPERPARAMETRI NON AGGIORNO LE MATRICI A, Q E P MA SOLO LE LORO DERIVATE!!
        self.N = np.size(observation)
        self.time = time
        self.y = np.array(observation)
        self.A = np.array(A)
        self.H = np.array(H)
        self.P0 = np.array(P0)
        self.Pinf = np.array(Pinf)
        self.Q = np.array(Q)
        self.m_x = np.zeros([self.N, 2])
        self.P_x = np.zeros([2, 2, self.N])

        self.time_steps = np.size(observation)

        self.lengthscale = hyperparameters[0]**2
        self.variance = hyperparameters[1]**2
        self.m_bar = observation
        self.C_bar = np.identity(self.time_steps) * self.variance
        self.y_hat = np.random.randn(self.N)

        self.B = np.identity(self.N)

        # Natural parameters
        self.nat_param_bar_1 = inv(self.C_bar) @ self.m_bar
        self.nat_param_bar_2 = -0.5 * inv(self.C_bar)

        # Learning rate
        self.lr = 1e-3
        self.func_count = 0

        self.KF_solver = SR1_solver(self.N)
        # questi dopo mi sa...
        # self.m_x(:, 1) = sol
        # self.m_x(:, 2) = randn(TRAIN_POINTS, 1);
        # self.P_x = zeros(2, 2, TRAIN_POINTS);
        # self.P_x(:,:,1) = P0;
        # self.x_hat = zeros(TRAIN_POINTS, 2);
        # self.P_hat = zeros(2, 2, TRAIN_POINTS);

        # Initialize plot
        self.fig = go.Figure()
        self.fig.add_trace(go.Line(x=time, y=self.y_hat, name='Estimated'))
        self.fig.add_trace(go.Line(x=time, y=self.y, name='Real'))
        self.fig.show()
        

    def filter(self):
        logZ = 0.0
        le = 0.0
        self.m_x = np.random.randn(self.time_steps, 2)
        self.P_x[:,:,0] = self.P0
        for t in range(2, self.time_steps):
            self.m_x[t, :] = self.A @ self.m_x[t-1, :].T      # m_n short bar in the paper
            self.P_x[:, :, t] = self.A @ self.P_x[:,:,t-1] @ self.A.T + self.Q
            
            # log(Z)
            S = self.H @ (self.A @ self.P_x[:, :, t-1] @ self.A.T + self.Q) @ self.H.T + self.C_bar[t,t]
            # dS_dlam_h = lambda lam, sigma : self.H @ (dA_dlam_h(lam, sigma) * P_x(:, :, t-1) * A' + A * P_x(:, :, t-1) * dA_dlam_h(lam, sigma)' + dQ_dlam_h(lam, sigma)) * H';
            logZ = logZ + np.log(norm.pdf(self.m_bar[t], self.H @ self.m_x[t, :].T, S))
            # Laplace energy
            le = le - np.log(norm.pdf(self.m_bar[t], self.H @ self.m_x[t, :].T, self.variance)) + np.log(norm.pdf(self.H @ self.m_x[t, :].T, self.m_bar[t], self.C_bar[t, t])) - logZ
        
            V = self.H @ self.P_x[:, :, t] @ self.H.T + self.C_bar[t,t]
            W = self.P_x[:, :, t] @ self.H.T @ inv(V)
            self.m_x[t, :] = self.m_x[t, :].T + W @ (self.m_bar[t] - self.H @ self.m_x[t,:].T)
            self.P_x[:, :, t] = self.P_x[:, :, t] - W @ V @ W.T

    def smoother(self):
        self.x_hat = self.m_x
        self.P_hat = self.P_x
        for t in range(self.time_steps-1, 0, -1):
            G = self.P_x[:, :, t] @ self.A @ inv(self.P_x[:, :, t])
            R = self.A @ self.P_x[:, :, t] @ self.A.T + self.Q
            self.m_x[t, :] = self.m_x[t, :].T + G @ (self.m_x[t+1,:].T - self.A @ self.m_x[t,:].T)
            self.P_x[:, :, t] = self.P_x[:, :, t] + G @ (self.P_x[:, :, t+1] - R) @ G.T
            # x_hat(t, :) = H * m_x(t, :)';
            # P_hat(:, :, t) = H * P_x(:, :, t) * H';
            self.x_hat[t, :] = self.m_x[t, :].T
            self.P_hat[:, :, t] = self.P_x[:, :, t]
        
        self.y_hat = (self.H @ self.x_hat.T).T  # this is m_n

    def update_nat_params(self):
        # Update the approximate likelihood by computing Jacobian and Hessian
        _, df = self.surrogate_fun(self.y, self.y_hat, self.variance)               # Compute the Jacobian of the surrogate target (LogLH in this case) wrt to estimated y
        self.func_count = self.func_count + 1
        
        # Stepping with a certain learning rate
        self.nat_param_bar_2 = (1 - self.lr) * self.nat_param_bar_2 + self.lr * 0.5 * self.B
        self.nat_param_bar_1 = (1 - self.lr) * self.nat_param_bar_1 + self.lr * (df - self.B * self.y_hat)
        
        # Convert approximate likelihood natural parametrs back to mean and
        # covariance
        self.C_bar = -0.5 * inv(self.nat_param_bar_2)
        self.m_bar = self.C_bar @ self.nat_param_bar_1

    def surrogate_fun(self, f: npt.NDArray, y: npt.NDArray, variance: float) -> tuple[float, npt.NDArray]:
        # The surrogate function in this case is the log Likelihood function and its derivative wrt f
        # m is m_(k,n) in the paper
        out = np.sum( 0.5 * (-np.log(variance) + (y - f) / variance * (y - f) - np.log(2 * np.pi)) )
        # dout = zeros(length(y), 1);
        dout_df = -2 * (y - f) * variance
        return out, dout_df
    
    def symmetric_rank1_update(self):
        # Update matrix B using SR1
        LogLH = lambda x : self.surrogate_fun(x, self.y, self.variance)
        _, self.B = self.KF_solver.step_forward(cost_function=LogLH, diff_wrt=0)

    def run(self):
        self.filter()
        self.smoother()
        self.update_nat_params()
        # self.update_plot()
        self.symmetric_rank1_update()


    def laplace_energy(self):
        pass

    def dlaplace_energy(self):
        pass