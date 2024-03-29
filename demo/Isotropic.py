"""
Homogeneous turbulence. See [1] for initialization and [2] for a section
on forcing the lowest wavenumbers to maintain a constant turbulent
kinetic energy.

[1] R. S. Rogallo, "Numerical experiments in homogeneous turbulence,"
NASA TM 81315 (1981)

[2] A. G. Lamorgese and D. A. Caughey and S. B. Pope, "Direct numerical simulation
of homogeneous turbulence with hyperviscosity", Physics of Fluids, 17, 1, 015106,
2005, (https://doi.org/10.1063/1.1833415)

"""

from __future__ import print_function
import warnings
import h5py # Reference: http://docs.h5py.org/en/stable/
import numpy as np
from numpy import pi, zeros, sum
#https://shenfun.readthedocs.io/en/latest/
from shenfun import Function, Array
from shenfun.fourier import energy_fourier
from spectralDNS import config, get_solver, solve
# Paraview I/O:
from mpi4py_fft import generate_xdmf

#################################   WARNINGS   #################################

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

################################   FUNCTIONS   ################################
#
#   Functions:
#       - initialize()
#       - L2_norm()
#       - spectrum()
#       - update()
#       - init_from_file()

def initialize(solver, context):
    c = context
    # Create mask with ones where |k| < Kf2 and zeros elsewhere
    kf = config.params.Kf2
    c.k2_mask = np.where(c.K2 <= kf**2, 1, 0)
    np.random.seed(solver.rank)
    # print(solver.rank) -> 0 (, or 1, 2, ...)
    k = np.sqrt(c.K2)
    k = np.where(k == 0, 1, k)
    kk = c.K2.copy()
    kk = np.where(kk == 0, 1, kk)
    k1, k2, k3 = c.K[0], c.K[1], c.K[2]
    ksq = np.sqrt(k1**2+k2**2)
    ksq = np.where(ksq == 0, 1, ksq)

    E0 = np.sqrt(9./11./kf*c.K2/kf**2)*c.k2_mask
    E1 = np.sqrt(9./11./kf*(k/kf)**(-5./3.))*(1-c.k2_mask)
    Ek = E0 + E1
    # theta1, theta2, phi, alpha and beta from [1]
    theta1, theta2, phi = np.random.sample(c.U_hat.shape)*2j*np.pi
    alpha = np.sqrt(Ek/4./np.pi/kk)*np.exp(1j*theta1)*np.cos(phi)
    beta = np.sqrt(Ek/4./np.pi/kk)*np.exp(1j*theta2)*np.sin(phi)
    c.U_hat[0] = (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    c.U_hat[1] = (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    c.U_hat[2] = beta*ksq/k
    c.mask = c.T.get_mask_nyquist()
    c.T.mask_nyquist(c.U_hat, c.mask)

    solver.get_velocity(**c)
    U_hat = solver.set_velocity(**c)

    K = c.K
    # project to zero divergence
    U_hat[:] -= (K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2])*c.K_over_K2

    if solver.rank == 0:
        c.U_hat[:, 0, 0, 0] = 0.0

    # Scale to get correct kinetic energy. Target from [2]
    energy = 0.5*energy_fourier(c.U_hat, c.T)
    target = config.params.Re_lam*(config.params.nu*config.params.kd)**2/np.sqrt(20./3.)
    c.U_hat *= np.sqrt(target/energy)

    if 'VV' in config.params.solver:
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    config.params.t = 0.0
    config.params.tstep = 0
    c.target_energy = energy_fourier(c.U_hat, c.T)

def L2_norm(comm, u):
    r"""Compute the L2-norm of real array a

    Computing \int abs(u)**2 dx

    """
    N = config.params.N
    result = comm.allreduce(np.sum(u**2))
    return result/np.prod(N)

def spectrum(solver, context):
    c = context
    uiui = np.zeros(c.U_hat[0].shape)
    uiui[..., 1:-1] = 2*np.sum((c.U_hat[..., 1:-1]*np.conj(c.U_hat[..., 1:-1])).real, axis=0)
    uiui[..., 0] = np.sum((c.U_hat[..., 0]*np.conj(c.U_hat[..., 0])).real, axis=0)
    uiui[..., -1] = np.sum((c.U_hat[..., -1]*np.conj(c.U_hat[..., -1])).real, axis=0)
    uiui *= (4./3.*np.pi)

    # Create bins for Ek
    Nb = int(np.sqrt(sum((config.params.N/2)**2)/3))
    bins = np.array(range(0, Nb))+0.5
    z = np.digitize(np.sqrt(context.K2), bins, right=True)

    # Sample
    Ek = np.zeros(Nb)
    ll = np.zeros(Nb)
    for i, k in enumerate(bins[1:]):
        k0 = bins[i] # lower limit, k is upper
        ii = np.where((z > k0) & (z <= k))
        ll[i] = len(ii[0])
        Ek[i] = (k**3 - k0**3)*np.sum(uiui[ii])

    Ek = solver.comm.allreduce(Ek)
    ll = solver.comm.allreduce(ll)
    for i in range(Nb):
        if not ll[i] == 0:
            Ek[i] = Ek[i] / ll[i]

    E0 = uiui.mean(axis=(1, 2))
    E1 = uiui.mean(axis=(0, 2))
    E2 = uiui.mean(axis=(0, 1))

    ## Rij
    #for i in range(3):
    #    c.U[i] = c.FFT.ifftn(c.U_hat[i], c.U[i])
    #X = c.FFT.get_local_mesh()
    #R = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    ## Sample
    #Rii = np.zeros_like(c.U)
    #Rii[0] = c.FFT.ifftn(np.conj(c.U_hat[0])*c.U_hat[0], Rii[0])
    #Rii[1] = c.FFT.ifftn(np.conj(c.U_hat[1])*c.U_hat[1], Rii[1])
    #Rii[2] = c.FFT.ifftn(np.conj(c.U_hat[2])*c.U_hat[2], Rii[2])

    #R11 = np.sum(Rii[:, :, 0, 0] + Rii[:, 0, :, 0] + Rii[:, 0, 0, :], axis=0)/3

    #Nr = 20
    #rbins = np.linspace(0, 2*np.pi, Nr)
    #rz = np.digitize(R, rbins, right=True)
    #RR = np.zeros(Nr)
    #for i in range(Nr):
    #    ii = np.where(rz == i)
    #    RR[i] = np.sum(Rii[0][ii] + Rii[1][ii] + Rii[2][ii]) / len(ii[0])

    #Rxx = np.zeros((3, config.params.N[0]))
    #for i in range(config.params.N[0]):
    #    Rxx[0, i] = (c.U[0] * np.roll(c.U[0], -i, axis=0)).mean()
    #    Rxx[1, i] = (c.U[0] * np.roll(c.U[0], -i, axis=1)).mean()
    #    Rxx[2, i] = (c.U[0] * np.roll(c.U[0], -i, axis=2)).mean()

    return Ek, bins, E0, E1, E2

k = []
w = []
kold = zeros(1)
im1 = None
energy_new = None
def update(context, backcoupling=False, validate=False):
    #backcoupling is switched on if called by the corresponding solver
    #validate is switched on if the fully coupled solver is being validated
    global k, w, im1, energy_new
    c = context
    params = config.params
    solver = config.solver
    curl_hat = Function(c.VT, buffer=c.work[(c.U_hat, 2, True)])

    if solver.rank == 0 and backcoupling==False:
        c.U_hat[:, 0, 0, 0] = 0


    if params.solver == 'VV':
        c.U_hat = solver.cross2(c.U_hat, c.K_over_K2, c.W_hat)

    energy_new = energy_fourier(c.U_hat, c.T)
    energy_lower = energy_fourier(c.U_hat*c.k2_mask, c.T)
    energy_upper = energy_new - energy_lower

    alpha2 = (c.target_energy - energy_upper) /energy_lower
    beta2 = c.target_energy / (energy_upper + energy_lower)
    #print("target", c.target_energy)
    #print(alpha2)
    alpha = np.sqrt(alpha2)
    beta = np.sqrt(beta2)

    #du = c.U_hat*c.k2_mask*(alpha)
    #dus = energy_fourier(du*c.U_hat, c.T)

    energy_old = energy_new

    #c.dU[:] = alpha*c.k2_mask*c.U_hat
    #scale velocities below cutoff wave number

    if (validate==False and alpha2 >= 0) or backcoupling==False:
        c.U_hat *= (alpha*c.k2_mask + (1-c.k2_mask))

    #if upper wavenumbers have too much energy, downscale all wave numbers
    elif (validate==False):
        c.U_hat[:] *= beta

    energy_new = energy_fourier(c.U_hat, c.T)

    #print("energy new",energy_new)
    #print("target energy", c.target_energy)

    if validate==False:
        assert np.sqrt((energy_new-c.target_energy)**2) < 1e-7, np.sqrt((energy_new-c.target_energy)**2)

    if params.solver == 'VV':
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    if (params.tstep % params.compute_energy == 0 or
            params.tstep % params.plot_step == 0 and params.plot_step > 0):
        solver.get_velocity(**c)
        solver.get_curl(**c)
        if 'NS' in params.solver:
            solver.get_pressure(**c)

    K = c.K
    if plt is not None:
        if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
            #div_u = solver.get_divergence(**c)

            if not plt.fignum_exists(1):
                plt.figure(1)
                #im1 = plt.contourf(c.X[1][:,:,0], c.X[0][:,:,0], div_u[:,:,10], 100)
                im1 = plt.contourf(c.X[1][..., 0], c.X[0][..., 0], c.U[0, ..., 10], 100)
                plt.colorbar(im1)
                plt.draw()
            else:
                im1.ax.clear()
                #im1.ax.contourf(c.X[1][:,:,0], c.X[0][:,:,0], div_u[:,:,10], 100)
                im1.ax.contourf(c.X[1][..., 0], c.X[0][..., 0], c.U[0, ..., 10], 100)
                im1.autoscale()
            plt.pause(1e-6)
        # Added by Joan:
            str_animation = "NS_visualize/Isotropic_"+str(params.tstep)+".png"
            plt.savefig(str_animation)

    """
    if params.tstep % params.compute_spectrum == 0:
        Ek, _, _, _, _ = spectrum(solver, context)
        f = h5py.File(context.spectrumname, driver='mpio', comm=solver.comm)
        f['Turbulence/Ek'].create_dataset(str(params.tstep), data=Ek)
        f.close()
    """
    if params.tstep % params.compute_energy == 0:
        dx, L = params.dx, params.L
        #ww = solver.comm.reduce(sum(curl*curl)/np.prod(params.N)/2)

        duidxj = np.zeros(((3, 3)+c.U[0].shape), dtype=c.float)
        for i in range(3):
            for j in range(3):
                duidxj[i, j] = c.T.backward(1j*K[j]*c.U_hat[i], duidxj[i, j])

        ww2 = L2_norm(solver.comm, duidxj)*params.nu
        #ww2 = solver.comm.reduce(sum(duidxj*duidxj))

        ddU = np.zeros(((3,)+c.U[0].shape), dtype=c.float)
        dU = solver.ComputeRHS(c.dU, c.U_hat, solver, **c)
        for i in range(3):
            ddU[i] = c.T.backward(dU[i], ddU[i])

        ww3 = solver.comm.allreduce(sum(ddU*c.U))/np.prod(params.N)

        ##if solver.rank == 0:
            ##print('W ', params.nu*ww, params.nu*ww2, ww3, ww-ww2)
        curl_hat = solver.cross2(curl_hat, K, c.U_hat)
        dissipation = energy_fourier(curl_hat, c.T)
        div_u = solver.get_divergence(**c)
        #du = 1j*(c.K[0]*c.U_hat[0]+c.K[1]*c.U_hat[1]+c.K[2]*c.U_hat[2])
        div_u = L2_norm(solver.comm, div_u)
        #div_u2 = energy_fourier(solver.comm, 1j*(K[0]*c.U_hat[0]+K[1]*c.U_hat[1]+K[2]*c.U_hat[2]))

        kk = 0.5*energy_new
        eps = dissipation*params.nu
        Re_lam = np.sqrt(20*kk**2/(3*params.nu*eps))
        Re_lam2 = kk*np.sqrt(20./3.)/(params.nu*params.kd)**2

        kold[0] = energy_new
        e0, e1 = energy_new, L2_norm(solver.comm, c.U)
        ww4 = (energy_new-energy_old)/2/params.dt

        # Here is what is output on the screan:
        if solver.rank == 0:
            k.append(energy_new)
            w.append(dissipation)
            print('\nt = %2.4f\ne0 = %2.6e\ne1 = %2.6e\neps = %2.6e\nww2 = %2.6e\nww3 = %2.6e\nww4 = %2.6e\nRe_lam = %2.6e\nRe_lam2 = %2.6e\n'%(params.t, e0, e1, eps, ww2, ww3, ww4, Re_lam, Re_lam2))

    #if params.tstep % params.compute_energy == 1:
        #if 'NS' in params.solver:
            #kk2 = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx[0]*dx[1]*dx[2]/L[0]/L[1]/L[2]/2)
            #if rank == 0:
                #print 0.5*(kk2-kold[0])/params.dt

def init_from_file(filename, solver, context):
    f = h5py.File(filename, mode='r', driver="mpio", comm=solver.comm)
    assert "0" in f["U/3D"]
    U_hat = context.U_hat
    s = context.T.local_slice(True)
    U_hat[:] = f["U/3D/0"][:, s[0], s[1], s[2]]
    if solver.rank == 0:
        U_hat[:, 0, 0, 0] = 0.0

    if 'VV' in config.params.solver:
        context.W_hat = solver.cross2(context.W_hat, context.K, context.U_hat)

    context.target_energy = energy_fourier(U_hat, context.T)

    f.close()

###################################   MAIN   ###################################

#   Use of: 'if __name__ == "__main__":':
#   This is because we want to write a .py file that can be both used by other
#   programs and/or modules as a module, and can also be run as the main program
#   itself. If your script is being imported into another module, its various
#   function and class definitions will be imported and its top-level code will
#   be executed, but the code in the then-body of the if clause above won't get
#   run as the condition is not met.
#
#   Reference: https://stackoverflow.com/questions/419163/what-does-if-name-main-do
#
#   Then, the conditional runs just when we do '$ ... python Isotropic.py ...'.
#   However, if we do 'import Isotropic.py', the stuff inside the 'if' will not
#   run.

if __name__ == "__main__":
    #import h5py # Reference: http://docs.h5py.org/en/stable/

    # Configure the parameters of spectralDNS from config.py:
    #
    config.update(            # This is a function defined in config.py
        {'nu': 0.005428,              # Viscosity (not used, see below)
         'dt': 0.001,                 # Time step
         'T': 15,                      # End time
         'L': [2.*pi, 2.*pi, 2.*pi],
         'checkpoint': 1000,
         'write_result': 1e8,
         'dealias': '3/2-rule',
        }, "triplyperiodic"
    )

    # OBS:  k0 = 2*pi / L = 1   (because domain L = 2*pi)
    #       kf/k0 = 3           (because domain k0 = 1)
    #       kd/k0 = 50          (because domain k0 = 1)
    #       nu =  5.428e-03     (because domain k0 = 1)
    #       Re = 43             (because domain h = 1)
    #       Re_lam = 84         (because domain h = 1)
    #       h = hyperviscosity index
    #       k = turbulent kinetic energy

    config.triplyperiodic.add_argument("--N", default=[256, 256, 256], nargs=3,
                                       help="Mesh size. Trumps M.")
    config.triplyperiodic.add_argument("--compute_energy", type=int, default=100)
    config.triplyperiodic.add_argument("--compute_spectrum", type=int, default=1000)
    config.triplyperiodic.add_argument("--plot_step", type=int, default=5)
    config.triplyperiodic.add_argument("--Kf2", type=int, default=3)        # (specified) largest wavenumber acted upon by the forcing
    config.triplyperiodic.add_argument("--kd", type=float, default=50.)     #  hyperviscous dissipation wavenumber
    config.triplyperiodic.add_argument("--Re_lam", type=float, default=84.) # microscale Reynolds number, Re_lambda
    config.triplyperiodic.add_argument("--validate", type=bool, default=False)

    # OBS: functions are objects in python, so they can be passed as arguments to other functions.
    # Here, passing the function 'update()' defined above (in Isotropic.py):
    sol = get_solver(update=update, mesh="triplyperiodic")  # see __init__.py
    config.params.nu = (1./config.params.kd**(4./3.))
                # sol is a module

    #
    # Finish configuration of the spectralDNS solver.
    #print(config.params.T)

    context = sol.get_context()     # get_context() is in NS.py
        # context is a dictionary of the class AttributeDict
    initialize(sol, context)

    ### Not called at the beggining! Called afterwards when willing to use the
    ### file as initial condition (previous simulation required):
    #init_from_file("NS_isotropic_128_128_128_c_initBS5ad.h5", sol, context)
    #init_from_file("NS_isotropic_256_256_256_c_initBS5ad.h5", sol, context)

    context.hdf5file.filename = "NS_isotropic_{}_{}_{}".format(*config.params.N)

    # Data for the 'NS_isotropic_{}_{}_{}.h5' file:
    Ek, bins, E0, E1, E2 = spectrum(sol, context)
    context.spectrumname = context.hdf5file.filename+".h5"
    f = h5py.File(context.spectrumname, mode='w', driver='mpio', comm=sol.comm)
    f.create_group("Turbulence")
    f["Turbulence"].create_group("Ek")
    bins = np.array(bins)
    f["Turbulence"].create_dataset("bins", data=bins)
    f.close()

    ### PRINT ###
    if sol.rank == 0:
        pass
        #print(int(config.params.T * 1 / config.params.dt))

    # Iterative process in time to solve the homogeneous isotropic turbulent flow:
    solve(sol, context)     # see __init__.py

    # Visualization in Paraview:
    if sol.rank == 0:
        #print(type(context.U))
        #print(context.U)
        #print(context.U[0, 0, 0, 0])
        #print(context.U[1, 0, 0, 0])
        #print(context.U[2, 0, 0, 0])
        pass
        #generate_xdmf('NS_isotropic_128_128_128_w.h5')
        #generate_xdmf('NS_isotropic_256_256_256_w.h5')
