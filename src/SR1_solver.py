import numpy as np
import numpy.typing as npt
import torch
from torch import autograd


class SR1_solver:
    def __init__(self, N: int) -> None:
        # Initialize solver parameters to default
        self.iter = 0
        self.func_count = 0
        self.grad_err = 1
        self.max_radius = 100           # Max trust region radius
        self.radius = 1                 # Initial trust region radius
        self.solution_update = False
        self.rho = 0                    # Reduction ratio

        self.N = N
        self.B = np.identity(self.N)
        self.solution = np.random.randn(self.N)
        self.candidate_solution = np.zeros(self.N)

    def step_solver(self, f: float, df: npt.NDArray) -> None:
        self.df = df
        self.f = f
        eps_tol = min(0.5, np.sqrt(np.linalg.norm(df))) * np.linalg.norm(df)
        z = np.zeros([self.N, 1])
        r = np.array(df)
        d = -r
        subproblem_iter = 0
        self.delta_solution = np.zeros([self.N, 1])

        while subproblem_iter < 5:
            min_fun = np.inf
            if d.T @ self.B @ d <= 0:
                fun = lambda tau : df.T @ (z + tau * d) + 0.5 * (z + tau * d).T @ self.B @ (z + tau * d)
                polyn = np.array([ d.T * d, 2 * d.T * z, z.T * z - self.radius**2 ])
    #                 tau_sol = real(roots(polyn))
                tau_sol = np.real( (-polyn[2] + np.sqrt(polyn[2]**2 - 4 * polyn[1] * polyn[3])) / (2 * polyn[1]) )
                for ii in range(np.size(tau_sol)):
                    if fun(tau_sol(ii)) < min_fun:
                        tau_min = tau_sol(ii)
                        min_fun = fun(tau_sol(ii))

                self.delta_solution = z + tau_min * d
                subp_status = 'Negative curvature {' + str(subproblem_iter) + '}'
                # Reset radius so it doesn't get stuck
    #             radius = 1
    #             B = eye(TRAIN_POINTS)
                break
            
            alpha = r.T * r / (d.T * self.B * d)
            z = z + alpha * d
            if np.linalg.norm(z) > self.radius:
    #             fun = @(tau) df' * (z + tau * d) + 0.5 * (z + tau * d)' * B * (z + tau * d)
                polyn = np.array([ d.T * d, 2 * d.T * z, z.T * z - self.radius**2 ])
                tau_sol = np.real( (-polyn[2] + np.sqrt(polyn[2]**2 - 4 * polyn[1] * polyn[3])) / (2 * polyn[1]) )
                self.delta_solution = z + max(tau_sol) * d

                subp_status = 'z beyond boundary {' + str(subproblem_iter) + '}'
                break
            
            r_new = r + alpha * self.B * d
            if np.linalg.norm(r_new) < eps_tol:
                self.delta_solution = z
                subp_status = 'Degenerated in Newton method {' + str(subproblem_iter) + '}'
    #             radius = 1
    #             B = eye(length(x0)) * 1e1
                break
            
            beta = r_new.T @ r_new / (r.T @ r)
            d = -r_new + beta * d
            r = r_new
            subproblem_iter = subproblem_iter + 1
        

        self.candidate_solution = self.solution + self.delta_solution # Candidate next solution

    def update_solution(self, candidate_f: float, candidate_df: npt.NDArray) -> None:
        # Compute candidate function value and its derivative using self.candidate_solution...
        # [ feval_tmp, dfeval_tmp ] = f_handle(y, sol_tmp, exp(theta(2))^2);

        self.yk = candidate_df - self.df

        actual_red = self.f - candidate_f
        predicted_red = -(self.df.T @ self.delta_solution + 0.5 * self.delta_solution.T @ self.B @ self.delta_solution)

        self.rho = actual_red / predicted_red
        self.solution_update = False
        if self.rho > 1e-6:
            self.solution = self.candidate_solution
            # grad_err = norm(dfeval_tmp, 'inf');
            # self.f = candidate_f
            self.iter = self.iter + 1
            self.solution_update = True
        
    def update_radius(self):
        if self.rho < 0.15: #&& radius > 0.01
            self.radius = 0.9 * self.radius
        else:
            if self.rho > 3/4 and (np.linalg.norm(self.delta_solution) - self.radius) <= 1e-6:
                self.radius = min(1.5 * self.radius, self.max_radius)

        if self.radius < 1e-6 and self.solution_update == False:
            exit_msg = 'Trust region radius lower than tolerance. Stopped.'

    def update_hessian(self):
        condition_on_B = abs(self.delta_solution.T @ (self.yk - self.B @ self.delta_solution)) >= 1e-8 * np.linalg.norm(self.delta_solution) * np.linalg.norm(self.yk - self.B @ self.delta_solution)
        if condition_on_B:
            self.B = self.B + (self.yk - self.B @ self.delta_solution) @ (self.yk - self.B @ self.delta_solution).T / ( (self.yk - self.B @ self.delta_solution).T * self.delta_solution)
        