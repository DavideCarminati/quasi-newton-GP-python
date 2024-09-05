# shebang

import sys
import SR1_solver
import system
import markovGP
import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.io as pio

# pio.renderers.default = "svg"

def main():
    dyn_sys = system.System()
    y, y_noiseless, times, x = dyn_sys.simulate(st_dev_noise=np.sqrt(1e-3)*0)
    hyperparams = np.array([1.5, np.sqrt(1e-3)])
    A, H, P0, Pinf, Q = dyn_sys.matern32_to_lti(hyperparameters=hyperparams)

    # fig = px.line(x=times, y=y)
    # # fig.show(renderer='notebook_connected')
    # fig.write_image("plots/prova.svg")

    mgp = markovGP.MarkovGP(time=times, observation=y, A=A, H=H, P0=P0, Pinf=Pinf, Q=Q, hyperparameters=hyperparams)

    for _ in range(3):
        mgp.run()


if __name__ == "__main__":
    main()