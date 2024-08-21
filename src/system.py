import numpy as np

# I need scipy signals for lsim method!!!

# m = 1;
# k = 1;
# b = 0.2;

# A = [0 1; -k/m -b/m];
# B = [0 1/m]';
# C = [1 0];
# D = [0];

# sys = ss(A,B,C,D);

# TRAIN_POINTS = 1000;
# time = linspace(0, 15, TRAIN_POINTS);
# u_vet = zeros(size(time));
# % Integration
# [y_noiseless,times,x] = lsim(sys,u_vet,time, [1; 0]);

# % Add sensor noise
# st_dev_noise = sqrt(1e-3);
# y = y_noiseless + st_dev_noise * randn(length(x), 1);

# %% Define SDE matrices using Matern 3/2 kernel

# variance = st_dev_noise^2;
# lengthscale = 1.5;        % Cambiare questa!!!
# % kernel_to_state_space(self, R=None):
# lam = 3.0 ^ 0.5 / lengthscale;
# F = [0.0, 1.0;
#     -lam ^ 2, -2 * lam];
# L = [0; 1];
# % Q = [12.0 * 3.0 ^ 0.5 / lengthscale ^ 3.0 * variance]; % Actually this is spectral density of beta. We need Qn
# H = [1.0, 0.0];
# Pinf = [variance, 0.0;
#         0.0, 3.0 * variance / lengthscale ^ 2.0];
# P0 = Pinf;
# % state_transition(self, dt):
    
# %     Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
# %     :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
# %     :return: state transition matrix A [2, 2]
    
# lam = sqrt(3.0) / lengthscale;
# dt = times(2) - times(1);
# A = expm(-dt * lam) * (dt * [lam, 1.0; -lam^2.0, -lam] + eye(2));

# Q = Pinf - A * Pinf * A';