"""
Velocity-vorticity formulation

Derived by taking the curl of the momentum equation and solving for the curl.
The velocity that is required in the convective term is computed from the curl
in the compute_velocity method.

This solver inherits most features from the NS_shenfun solver.
Overloading just a few routines.

"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2018-10-23"
__copyright__ = "Copyright (C) 2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

from .NS import *

NS_context = get_context

def get_context():
    """Set up context for Velocity-Vorticity (VV) solver"""
    c = NS_context()
    del c.P, c.P_hat

    c.W_hat = Function(c.VT)

    # Primary variable
    c.u = c.W_hat
    c.hdf5file = VVFile(config.params.solver,
                        checkpoint={'space': c.VT,
                                    'data': {'0': {'curl': [c.W_hat]}}},
                        results={'space': c.VT,
                                 'data': {'U': [c.U], 'curl': [c.curl]}})

    return c

class VVFile(HDF5File):
    """Subclass HDF5File for appropriate updating of real components

    The method 'update_components' is used to transform all variables
    that are to be stored. If more variables than U and P are
    wanted, then subclass HDF5Writer in the application.
    """
    def update_components(self, **context):
        """Transform to real data before storing the solution"""
        get_velocity(**context)
        get_curl(**context)

def compute_velocity(U, w_hat, work, VT, K_over_K2):
    """Compute u from curl(u)

    Follows from
      w = [curl(u)=] \nabla \times u
      curl(w) = \nabla^2(u) (since div(u)=0)
      FFT(curl(w)) = FFT(\nabla^2(u))
      ik \times w_hat = k^2 u_hat
      u_hat = (ik \times w_hat) / k^2
      u = iFFT(u_hat)

    """
    v_hat = work[(w_hat, 1, True)]
    v_hat = cross2(v_hat, K_over_K2, w_hat)
    U = VT.backward(v_hat, U)
    return U

def get_velocity(W_hat, U, work, VT, K_over_K2, **context):
    """Compute velocity from context"""
    U = compute_velocity(U, W_hat, work, VT, K_over_K2)
    return U

def get_divergence(T, K, U_hat, W_hat, **context):
    div_u = Array(T)
    U_hat = cross2(U_hat, K, W_hat)
    div_u = T.backward(1j*(K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2]), div_u)
    return div_u

def get_curl(curl, W_hat, VT, **context):
    """Compute curl from context"""
    curl = VT.backward(W_hat, curl)
    return curl

def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):
        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, w_hat, work, Tp, VTp, K, K_over_K2, u_dealias):
            w_dealias = work[(u_dealias, 0, True)]
            v_hat = work[(rhs, 0, True)]

            u_dealias = compute_velocity(u_dealias, w_hat, work, VTp, K_over_K2)
            w_dealias = VTp.backward(w_hat, w_dealias)
            v_hat = Cross(v_hat, u_dealias, w_dealias, work, VTp) # v_hat = F_k(u_dealias x w_dealias)
            rhs = cross2(rhs, K, v_hat)  # rhs = 1j*(K x v_hat)
            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def add_linear(rhs, w_hat, nu, K2, Source):
    """Add contributions from source and diffusion to the rhs"""
    rhs -= nu*K2*w_hat
    rhs += Source
    return rhs

def ComputeRHS(rhs, w_hat, solver, work, Tp, VT, VTp, K, K2, K_over_K2,
               Source, u_dealias, mask, **context):
    """Return right hand side of Navier Stokes in velocity-vorticity form

    Parameters
    ----------
        rhs : array
            The right hand side to be returned
        w_hat : array
            The FFT of the curl at current time. May differ from
            context.W_hat since it is set by the integrator
        solver : module
            The solver module. Included for possible inheritance
            and flexibility of integrators.

    Other Parameters
    ----------------
        work : dict
            Work arrays
        Tp : TensorProductSpace
        K : list of arrays
            Scaled wavenumber mesh
        K2 : array
            K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
        K_over_K2 : array
            K / K2
        Source : array
            Scalar source term

    """
    rhs = solver.conv(rhs, w_hat, work, Tp, VTp, K, K_over_K2, u_dealias)
    if mask is not None:
        rhs.mask_nyquist(mask)
    rhs = solver.add_linear(rhs, w_hat, params.nu, K2, Source)
    return rhs
