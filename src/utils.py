import numpy as np
import numpy.typing as npt

def likelihood(y: npt.NDArray, f: npt.NDArray, variance: float) -> float:
    # Log Likelihood function and its derivative wrt f
    # m is m_(k,n) in the paper
    out = np.sum( 0.5 * (-np.log(variance) + (y - f) / variance * (y - f) - np.log(2 * np.pi)) )
    # dout = zeros(length(y), 1);
    # dout_f = -2 * (y - f) * variance
    return out

# def laplace_energy()