# shebang

import sys
import SR1_solver
import system
import markovGP
import numpy as np
import numpy.typing as npt
import plotly.express as px

def main():
    dyn_sys = system.System()
    y, y_noiseless, times, x = dyn_sys.simulate(st_dev_noise=1e-3)
    hyperparams = np.array([5, 0.1])
    A, H, P0, Pinf, Q = dyn_sys.matern32_to_lti(hyperparameters=hyperparams)

    fig = px.line(x=times, y=y)
    fig.show()


if __name__ == "__main__":
    main()