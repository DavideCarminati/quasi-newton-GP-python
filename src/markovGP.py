import numpy as np
from numpy.linalg import inv
import numpy.typing as npt
import scipy
import scipy.stats
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.scatter import Line
import plotly.io as pio
import time as tm

import autograd.scipy.stats.multivariate_normal as mvn
import autograd.numpy as anp

import utils
from SR1_solver import SR1_solver
from system import matern32_to_lti

# pio.renderers.default = "svg"


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
        self.dt = time[1] - time[0]
        self.y = np.array(observation)
        self.A = np.array(A)
        self.H = np.array(H)
        self.P0 = np.array(P0)
        self.Pinf = np.array(Pinf)
        self.Q = np.array(Q)
        self.m_x = np.random.rand(self.N, 2)
        self.P_x = np.zeros((2, 2, self.N))

        self.time_steps = np.size(observation)

        self.lengthscale = hyperparameters[0]
        self.variance = hyperparameters[1]**2
        self.m_bar = observation
        self.C_bar = np.identity(self.time_steps) * self.variance
        self.y_hat = np.random.randn(self.N)

        self.B = np.identity(self.N)

        # Natural parameters
        self.nat_param_bar_1 = inv(self.C_bar) @ self.m_bar
        # self.nat_param_bar_1 = 1 / self.variance * self.m_bar
        self.nat_param_bar_2 = -0.5 * inv(self.C_bar)

        # Learning rate
        self.lr = 1e-3
        self.func_count = 0

        self.KF_solver = SR1_solver(self.N)
        self.hyp_solver = SR1_solver(np.size(hyperparameters))

        # Initialize plot
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=time, y=self.y_hat, name='Estimated', mode='lines'))
        self.fig.add_trace(go.Scatter(x=time, y=self.y, name='Real', mode='lines'))
        self.fig.show()

        self.fig_hyp = go.Figure()
        tm.sleep(0.5)
        

    def filter(self, 
               m_x: npt.NDArray, 
               P_x: npt.NDArray) -> float:
        ''' 
        Computes filtered (posterior) mean and covariance of the states of the given dynamical system.

        :param m_x: array in which to write posterior mean
        :param P_x: array in which to write posterior covariance

        :return le: laplace energy
        '''
        logZ = 0.0
        le = 0.0
        self.m_x = np.random.randn(self.time_steps, 2)
        self.m_x[0, :] = [1, 0]
        self.P_x[:,:,0] = self.P0
        for t in range(1, self.time_steps):
            self.m_x[t, :] = self.A @ self.m_x[t-1, :].T      # m_n short bar in the paper
            self.P_x[:, :, t] = self.A @ self.P_x[:,:,t-1] @ self.A.T + self.Q
            
            # log(Z)
            S = self.H @ (self.A @ self.P_x[:, :, t-1] @ self.A.T + self.Q) @ self.H.T + self.C_bar[t,t]
            # dS_dlam_h = lambda lam, sigma : self.H @ (dA_dlam_h(lam, sigma) * P_x(:, :, t-1) * A' + A * P_x(:, :, t-1) * dA_dlam_h(lam, sigma)' + dQ_dlam_h(lam, sigma)) * H';
            logZ = logZ + np.log(norm.pdf(self.m_bar[t], self.H @ self.m_x[t, :].T, S))
            # Laplace energy
            le = le - np.log(norm.pdf(self.m_bar[t], self.H @ self.m_x[t, :].T, self.variance)) + np.log(norm.pdf(self.H @ self.m_x[t, :].T, self.m_bar[t], self.C_bar[t, t])) - logZ
        
            V = self.H @ self.P_x[:, :, t] @ self.H.T + self.C_bar[t,t]
            W = np.expand_dims(self.P_x[:, :, t] @ self.H.T / V, axis=-1) # Sarebbe @ inv(V) ma è una matrice 1x1
            self.m_x[t, :] = self.m_x[t, :].T + np.squeeze(W) * (self.m_bar[t] - self.H @ self.m_x[t,:].T) # Bello sto numpy @ (self.m_bar[t] - self.H @ self.m_x[t,:].T)
            self.P_x[:, :, t] = self.P_x[:, :, t] - V * W @ W.T # W @ V @ W.T

        return le

    def smoother(self):
        self.x_hat = self.m_x
        self.P_hat = self.P_x
        for t in range(self.time_steps-2, -1, -1):
            G = self.P_x[:, :, t] @ self.A @ inv(self.P_x[:, :, t])
            R = self.A @ self.P_x[:, :, t] @ self.A.T + self.Q
            self.m_x[t, :] = self.m_x[t, :].T + G @ (self.m_x[t+1,:].T - self.A @ self.m_x[t,:].T)
            self.P_x[:, :, t] = self.P_x[:, :, t] + G @ (self.P_x[:, :, t+1] - R) @ G.T
            # x_hat(t, :) = H * m_x(t, :)';
            # P_hat(:, :, t) = H * P_x(:, :, t) * H';
            self.x_hat[t, :] = self.m_x[t, :].T
            self.P_hat[:, :, t] = self.P_x[:, :, t]
        
        self.y_hat = (self.H @ self.x_hat.T).T  # this is m_n
        # self.y_hat = (self.H @ self.m_x.T).T

    def update_nat_params(self):
        # Update the approximate likelihood by computing Jacobian and Hessian
        _, df = self.surrogate_fun(self.y, self.y_hat, self.variance)               # Compute the Jacobian of the surrogate target (LogLH in this case) wrt to estimated y
        self.func_count = self.func_count + 1
        
        # Stepping with a certain learning rate
        self.nat_param_bar_2 = (1 - self.lr) * self.nat_param_bar_2 + self.lr * 0.5 * self.B
        self.nat_param_bar_1 = (1 - self.lr) * self.nat_param_bar_1 + self.lr * (df - self.B @ self.y_hat)
        
        # Convert approximate likelihood natural parametrs back to mean and
        # covariance
        self.C_bar = -0.5 * inv(self.nat_param_bar_2)
        self.m_bar = self.C_bar @ self.nat_param_bar_1

    def surrogate_fun(self, y: npt.NDArray, f: npt.NDArray, variance: float) -> tuple[float, npt.NDArray]:
        # The surrogate function in this case is the log Likelihood function and its derivative wrt f
        # m is m_(k,n) in the paper
        out = np.sum( 0.5 * (-np.log(variance) + (y - f) / variance * (y - f) - np.log(2 * np.pi)), axis=None )
        # dout = zeros(length(y), 1);
        dout_df = -2 * (y - f) * variance
        return out, dout_df
    
    def symmetric_rank1_update(self):
        # Update matrix B using SR1
        LogLH = lambda x : self.surrogate_fun(self.y, x, self.variance)
        _, self.B = self.KF_solver.step_forward(self.y_hat, cost_fun_value_and_derivative=LogLH)

    def run(self):
        self.filter(self.m_x, self.P_x)
        self.smoother()
        self.update_nat_params()
        # self.update_plot()
        self.symmetric_rank1_update()

        # Now update hyperparamters
        self.update_hyperparameters()

        self.fig.update_traces(y=self.y_hat, selector = ({'name':'Estimated'}))
        self.fig.show()
        tm.sleep(1)


    def laplace_energy(self, hyperparameters: npt.NDArray) -> tuple[float, npt.NDArray]:
        '''Compute Laplace Energy and its derivatives wrt hyperparameters
        
        '''
        A, H, P0, Pinf, Q = matern32_to_lti(hyperparameters, self.dt)
        variance = hyperparameters[1]**2
        logZ = 0.0
        dlogZ_dlam = 0
        dlogZ_dsigma = 0
        dLE_dlam = 0
        dLE_dsigma = 0
        le = 0.0
        m_x = np.random.randn(self.time_steps, 2)
        P_x = np.zeros((2, 2, self.N))
        m_x[0, :] = np.array([1, 0])
        P_x[:,:,0] = P0
        for t in range(1, self.time_steps):

            m_x[t, :] = A @ m_x[t-1, :].T      # m_n short bar in the paper
            P_x[:, :, t] = A @ P_x[:,:,t-1] @ A.T + Q
            
            # log(Z)
            S = H @ (A @ P_x[:, :, t-1] @ A.T + Q) @ H.T + self.C_bar[t,t]
            S = (S + S.T)/2
            dS_dlam_h = lambda lam, sigma : self.H @ (self.dA_dlam_h(lam) @ P_x[:, :, t-1] @ self.A.T + self.A @ P_x[:, :, t-1] @ self.dA_dlam_h(lam).T + self.dQ_dlam_h(lam, sigma)) @ self.H.T
            logZ = logZ + np.log(mvn.pdf(self.m_bar[t], H @ m_x[t, :].T, S))
            # dlogZ/dlam
            dA_dlam = self.dA_dlam_h(hyperparameters[0])
            dS_dlam = dS_dlam_h(hyperparameters[0], hyperparameters[1])
            tmp = 0.5 * ( -(self.m_bar[t] - self.H @ dA_dlam @ m_x[t-1, :].T).T / S * (self.m_bar[t] - self.H @ self.m_x[t, :].T) \
                + (self.m_bar[t] - self.H @ m_x[t, :].T).T / S * dS_dlam / S * (self.m_bar[t] - self.H @ m_x[t, :].T) - \
                (self.m_bar[t] - self.H @ m_x[t, :].T).T / S * (self.m_bar[t] - self.H @ dA_dlam @ m_x[t-1, :].T) )
            dlogZ_dlam = dlogZ_dlam - 0.5 * (dS_dlam / S) + tmp
            dLE_dlam = dlogZ_dlam
            # dlogZ/dsigma
            dS_dsigma_h = lambda lam, sigma: self.H @ self.dQ_dsigma_h(lam, sigma) @ self.H.T
            dS_dsigma = dS_dsigma_h(hyperparameters[0], hyperparameters[1])
            dlogZ_dsigma = dlogZ_dsigma - 0.5 * (dS_dsigma / S) + \
                0.5 * (self.m_bar[t] - self.H @ m_x[t, :].T).T * (dS_dsigma / S / S) * (self.m_bar[t] - self.H @ m_x[t, :].T)
            # dLH_dsigma: derivative of measurement model (aka likelihood)
            dLH_dsigma = 1 /  hyperparameters[1] - (self.m_bar[t] - self.H @ m_x[t, :].T)**2 /  hyperparameters[1]**3
            dLE_dsigma = dLE_dsigma - dLH_dsigma - dlogZ_dsigma
            # Laplace energy
            le = le - np.log(norm.pdf(self.m_bar[t], H @ m_x[t, :].T, variance)) + np.log(norm.pdf(H @ m_x[t, :].T, self.m_bar[t], self.C_bar[t, t])) - logZ
        
            V = H @ P_x[:, :, t] @ H.T + self.C_bar[t,t]
            W = np.expand_dims(P_x[:, :, t] @ H.T / V, axis=-1) # Sarebbe @ inv(V) ma è una matrice 1x1
            m_x[t, :] = m_x[t, :].T + np.squeeze(W) * (self.m_bar[t] - H @ m_x[t,:].T) # Bello sto numpy @ (self.m_bar[t] - self.H @ self.m_x[t,:].T)
            P_x[:, :, t] = P_x[:, :, t] - V * W @ W.T # W @ V @ W.T

        dLE = np.array([dLE_dlam, dLE_dsigma])

        return le, dLE

    def update_hyperparameters(self):
        #
        LE = lambda x: self.laplace_energy(x)
        current_hyperparams = np.array([np.sqrt(3) / self.lengthscale, np.sqrt(self.variance)]) # Hyperparams are [lambda = sqrt(3)/l, sigma]
        new_hyperparameters, _ = self.hyp_solver.step_forward(current_hyperparams, cost_fun_value_and_derivative=LE)
        # Update new hyperparameters and system matrices
        self.lengthscale = np.sqrt(3) / new_hyperparameters[0]
        self.variance = new_hyperparameters[1]**2
        print("hyp: " + str(new_hyperparameters))
        self.A, self.H, self.P0, self.Pinf, self.Q = matern32_to_lti(new_hyperparameters, self.dt)

    ###### DERIVATIVES OF MATRICES WRT LAMBDA ######
    def dP0_dlam_h(self, lam: float, sigma: float) -> npt.NDArray:
        return np.array([[0, 0], [0, 2 * sigma * lam]])
    
    def dA_dlam_h(self, lam: float) -> npt.NDArray:
        A = np.array([[lam, 1], [-lam**2, -lam]])
        dA = np.array([[1, 0], [-2 * lam, -1]])
        return -self.dt * np.exp(-self.dt * lam) * (self.dt * A + np.identity(2)) + np.exp(-self.dt * lam) * (self.dt * dA)
    
    def dQ_dlam_h(self, lam: float, sigma: float) -> npt.NDArray:
        return self.dP0_dlam_h(lam, sigma) - self.dA_dlam_h(lam) @ self.P0 @ self.A.T - \
                self.A @ self.dP0_dlam_h(lam, sigma) @ self.A.T - self.A @ self.P0 @ self.dA_dlam_h(lam).T
    

    ###### DERIVATIVES OF MATRICES WRT SIGMA ######
    def dP0_dsigma_h(self, lam: float) -> npt.NDArray:
        return np.array([[1, 0], [0, lam**2]])
    
    def dQ_dsigma_h(self, lam: float , sigma: float) -> npt.NDArray:
        return self.dP0_dsigma_h(lam) - self.A @ self.dP0_dsigma_h(lam) @ self.A.T


# This takes too much time to run due to autograd...
    # def laplace_energy(self, 
    #                    hyperparameters: npt.NDArray):
    #     #
    #     A, H, P0, Pinf, Q = matern32_to_lti(hyperparameters, self.dt)
    #     variance = hyperparameters[1]**2
    #     logZ = 0.0
    #     le = 0.0
    #     # m_x = anp.random.randn(self.time_steps, 2)
    #     # P_x = anp.zeros((2, 2, self.N))
    #     m_x = np.array([1, 0])
    #     P_x = np.identity(2)
    #     for t in range(1, self.time_steps):
    #         # if t == 1:
    #         #     m_x = A @ m_x.T      # m_n short bar in the paper
    #         #     P_x = A @ P0 @ A.T + Q
                
    #         #     # log(Z)
    #         #     S = H @ (A @ P0 @ A.T + Q) @ H.T + self.C_bar[t,t]
    #         #     # dS_dlam_h = lambda lam, sigma : self.H @ (dA_dlam_h(lam, sigma) * P_x(:, :, t-1) * A' + A * P_x(:, :, t-1) * dA_dlam_h(lam, sigma)' + dQ_dlam_h(lam, sigma)) * H';
    #         #     logZ = logZ + anp.log(mvn.pdf(self.m_bar[t], H @ m_x.T, S))
    #         #     # Laplace energy
    #         #     le = le - anp.log(mvn.pdf(self.m_bar[t], H @ m_x.T, variance)) + anp.log(mvn.pdf(H @ m_x.T, self.m_bar[t], self.C_bar[t, t])) - logZ
            
    #         #     V = H @ P0 @ H.T + self.C_bar[t,t]
    #         #     W = anp.expand_dims(P0 @ H.T / V, axis=-1) # Sarebbe @ inv(V) ma è una matrice 1x1
    #         #     m_x = m_x.T + anp.squeeze(W) * (self.m_bar[t] - H @ m_x.T) # Bello sto numpy @ (self.m_bar[t] - self.H @ self.m_x[t,:].T)
    #         #     P_x = P0 - V * W @ W.T # W @ V @ W.T
    #         # else:
    #             m_x = A @ m_x.T      # m_n short bar in the paper
    #             P_x = A @ P_x @ A.T + Q
                
    #             # log(Z)
    #             S = H @ (A @ P_x @ A.T + Q) @ H.T + self.C_bar[t,t]
    #             # dS_dlam_h = lambda lam, sigma : self.H @ (dA_dlam_h(lam, sigma) * P_x(:, :, t-1) * A' + A * P_x(:, :, t-1) * dA_dlam_h(lam, sigma)' + dQ_dlam_h(lam, sigma)) * H';
    #             logZ = logZ + anp.log(mvn.pdf(self.m_bar[t], H @ m_x.T, S))
    #             # Laplace energy
    #             le = le - anp.log(mvn.pdf(self.m_bar[t], H @ m_x.T, variance)) + anp.log(mvn.pdf(H @ m_x.T, self.m_bar[t], self.C_bar[t, t])) - logZ
            
    #             V = H @ P_x @ H.T + self.C_bar[t,t]
    #             W = anp.expand_dims(P_x @ H.T / V, axis=-1) # Sarebbe @ inv(V) ma è una matrice 1x1
    #             m_x = m_x.T + anp.squeeze(W) * (self.m_bar[t] - H @ m_x.T) # Bello sto numpy @ (self.m_bar[t] - self.H @ self.m_x[t,:].T)
    #             P_x = P_x - V * W @ W.T # W @ V @ W.T

    #     return le