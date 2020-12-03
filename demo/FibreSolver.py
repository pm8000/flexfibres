"""
Fibre dynamics solver
Joan Clapes Roig
"""

from __future__ import print_function
from spectralDNS import config
from spectralDNS.utilities import Timer
from fibredyn import fibutils
import numpy as np
#import pdb     # -> Python Debugger

def get_fibcontext():
    """ Get the context of the fibre solver, define the arrays and
    variables necessary to solve the fibre dynamics.

    The context is an object of the class AttributeDict, which inherits
    from the dict class and defines several dunder methods.

    Returns a dictionary that allows to access each local variable defined
    in get_fibcontext(). If we do fib_context = get_fibcontext(), then we can
    access each variable as e.g. fib_context.xdotdot_old and perform any
    modification on its value. """

    n = config.params.fib_n
    n_plus = config.params.fib_n_plus

    # Recall: i = 0 is the spider, i = 0, 1, ..., n

    # All the variables are an object of type ndarray (numpy). The scalars are
    # stored for every mass point as a 1D array of length the number of mass
    # points (=n_plus), and the vectors are stored for every mass point as a 2D
    # array of shape the number of mass points (=n_plus) x 3.

    # z, gfun and h correspond to the extended variables and functions defined
    # in the report in order to carry out time integration. Each component is
    # associated with a nonlinear ODE, in total 6*n_plus

    # Mass array:
    m = np.array([config.params.fib_m0] + n*[config.params.fib_m_other])

    # Variables for the old time step:
    x_old = np.empty((n_plus, 3))
    xdot_old = np.empty((n_plus, 3))
    xdotdot_old = np.empty((n_plus, 3))
    z_old = np.zeros(n_plus*6)
    gfun_old = np.zeros(n_plus*6)
    h_old = np.zeros(n_plus*6)

    # Variables for the new time step:
    x_new = np.zeros((n_plus, 3))
    xdot_new = np.zeros((n_plus, 3))
    xdotdot_new = np.zeros((n_plus, 3))
    z_new = np.zeros(n_plus*6)
    gfun_new = np.zeros(n_plus*6)
    h_new = np.zeros(n_plus*6)

    # Intermediate variables:
    r = np.zeros((n_plus, 3))
    r_abs = np.zeros(n_plus)
    e_r = np.zeros((n_plus, 3))
    e_n = np.zeros((n_plus, 3))
    rdot = np.zeros((n_plus, 3))
    ang = np.zeros(n_plus)
    angdot = np.zeros((n_plus, 3))
    kappa = np.zeros((n_plus, 3))
    kappadot = np.zeros((n_plus, 3))
    s = np.zeros(n_plus)
    eps = np.zeros(n_plus)
    epsdot = np.zeros(n_plus)
    F_k = np.zeros((n_plus, 3))
    F_d = np.zeros((n_plus, 3))
    M_b = np.zeros((n_plus, 3))
    M_c = np.zeros((n_plus, 3))
    F = np.zeros((n_plus, 3))
    M = np.zeros((n_plus, 3))
    Q = np.zeros((n_plus, 3))
    D_n = np.zeros((n_plus, 3))
    D_l = np.zeros((n_plus, 3))
    D = np.zeros((n_plus, 3))
    G = np.zeros((n_plus, 3))
    cross_e = np.zeros((n_plus, 3))
    cross_e_abs = np.zeros(n_plus)
    v_rel = np.zeros((n_plus, 3))
    u1 = np.zeros((n_plus, 3))
    u2 = np.zeros((n_plus, 3))
    u3 = np.zeros((n_plus, 3))
    u_ip = np.zeros((n_plus, 3))
    Re = []     # list that will contain all the Reynolds numbers
    nfev = []

    # Variables for the iterative method:
    _xdotdot = n_plus*[0]
    _r = n_plus*[0]
    _r_abs = n_plus*[0]
    _e_r = n_plus*[0]
    _e_n = n_plus*[0]
    _rdot = n_plus*[0]
    _ang = n_plus*[0]
    _angdot = n_plus*[0]
    _kappa = n_plus*[0]
    _kappadot = n_plus*[0]
    _s = n_plus*[0]
    _eps = n_plus*[0]
    _epsdot = n_plus*[0]
    _F_k = n_plus*[0]
    _F_d = n_plus*[0]
    _M_b = n_plus*[0]
    _M_c = n_plus*[0]
    _F = n_plus*[0]
    _M = n_plus*[0]
    _Q = n_plus*[0]
    _D_n = n_plus*[0]
    _D_l = n_plus*[0]
    _D = n_plus*[0]
    _G = n_plus*[0]
    _cross_e = n_plus*[0]
    _cross_e_abs = n_plus*[0]
    _v_rel = n_plus*[0]
    _gfun = 6*n_plus*[0]
    _h = 6*n_plus*[0]

    return config.AttributeDict(locals())


def initialize_fibre(context, start=[0, 0, 0], axis=1, vel_ini = 0., axis_vel = 3):
    """ Initialize the positions, velocities and accelerations of the
    mass points of the fiber, as well as the other variables defined in
    the context. """

    c = context
    n = config.params.fib_n
    n_plus = config.params.fib_n_plus
    test = config.params.fib_test

    if test == "lindyn":
        w_forced = config.params.fib_w_forced
        t_F = config.params.t
        #F_ampl = config.params.fib_BC_Fz
        #Fz_dyn = F_ampl *(1 + np.cos(w_forced * t_F))
        F_ampl = config.params.fib_BC_Fz
        Fz_dyn = F_ampl * np.sin(w_forced * t_F)

    ## Initialize positions and velocities (add more possibilities):
    # The 3D domain is 2*pi x 2*pi x 2*pi
    if config.params.coupled_solver:
        fibutils.fib_orientation_init(c, start, axis, vel_ini, axis_vel)

    else:   # fiber dynamics solver alone
        if test == "linstat":
            start_pos_x1 = 0
            start_pos_x2 = 0
            start_pos_x3 = 0
        elif test == "lindyn":
            start_pos_x1 = 0
            start_pos_x2 = 0
            start_pos_x3 = 0
        elif test == "bend":
            start_pos_x1 = 0
            start_pos_x2 = 0
            start_pos_x3 = 0
        else:
            start_pos_x1 = 2 * np.pi / 4
            start_pos_x2 = 2 * np.pi / 2
            start_pos_x3 = 2 * np.pi / 2

        for i in range(n_plus):
            # Initialize the positions of the fiber:
            if test == "linstat":
                c.x_old[i, 0] = start_pos_x1
                c.x_old[i, 1] = start_pos_x2
                c.x_old[i, 2] = start_pos_x3 + i*config.params.fib_L0
            elif test == "lindyn":
                c.x_old[i, 0] = start_pos_x1
                c.x_old[i, 1] = start_pos_x2
                c.x_old[i, 2] = start_pos_x3 + i*config.params.fib_L0
            elif test == "bend":
                c.x_old[i, 0] = start_pos_x1 + i*config.params.fib_L0
                c.x_old[i, 1] = start_pos_x2
                c.x_old[i, 2] = start_pos_x3
            else:
                c.x_old[i, 0] = start_pos_x1 + i*config.params.fib_L0
                c.x_old[i, 1] = start_pos_x2
                c.x_old[i, 2] = start_pos_x3
            # Initialize velocities to zero:
            c.xdot_old[i, 0] = 0
            c.xdot_old[i, 1] = 0
            c.xdot_old[i, 2] = 0
            # Initialize the components of gfun that correspond to the velocities:
            c.gfun_old[3*i] = c.xdot_old[i, 0]
            c.gfun_old[3*i+1] = c.xdot_old[i, 1]
            c.gfun_old[3*i+2] = c.xdot_old[i, 2]
            # Initialize the components of the z vector:
            c.z_old[3*i] = c.x_old[i, 0]
            c.z_old[3*i+1] = c.x_old[i, 1]
            c.z_old[3*i+2] = c.x_old[i, 2]
            c.z_old[n_plus*3 + 3*i] = c.xdot_old[i, 0]
            c.z_old[n_plus*3 + 3*i+1] = c.xdot_old[i, 1]
            c.z_old[n_plus*3 + 3*i+2] = c.xdot_old[i, 2]

        # Note how the z vector and the gfun are defined: the first 3 components
        # ([0], [1], [2]) correspond to the position vector of the mass point 0,
        # the next three ([3], [4], [5]) correspond to the position vector of
        # the mass point 1, and so on, until the component [3*n_plus] is reached
        # and then the components ([3*n_plus], [3*n_plus+1], [3*n_plus+2])
        # correspond to the velocity vector of the mass point 0, the next three
        # ([3*n_plus+3], [3*n_plus+4], [3*n_plus+5]) correspond to the velocity
        # vector of the mass point 1, and so on. There are a total of n_plus
        # mass points, i=0,1,...,n.

    # Following Hostettler's notation presented in the report (note that not
    # all the variables really have n_plus components, but for convenience all
    # the variables were given size n_plus and then only the corresponding
    # components by index i are calculated, and the rest of the unnecessary
    # components are given the value of zero):

    c.G[0, ...] = fibutils.func_G(c.m[0])         # the spider's grav force
    # loop over all the mass points i = 1, ..., n (all of them without the spider)
    for i in range(1, n_plus):
        c.r[i, ...] = fibutils.func_r(c.x_old[i, ...], c.x_old[i-1, ...])
        c.r_abs[i] = fibutils.func_r_abs(c.r[i, ...])
        c.e_r[i, ...] = fibutils.func_e_r(c.r[i, ...], c.r_abs[i])
        c.rdot[i, ...] = fibutils.func_rdot(c.xdot_old[i, ...], c.xdot_old[i-1, ...])
        c.angdot[i, ...] = fibutils.func_angdot(c.e_r[i, ...], c.rdot[i, ...], c.r_abs[i])
        c.eps[i] = fibutils.func_eps(c.r_abs[i])
        c.epsdot[i] = fibutils.func_epsdot(c.e_r[i, ...], c.rdot[i, ...])

        c.F_k[i, ...] = fibutils.func_F_k(c.eps[i], c.e_r[i, ...])
        c.F_d[i, ...] = fibutils.func_F_d(c.epsdot[i], c.e_r[i, ...])
        c.F[i, ...] = fibutils.func_F(c.F_k[i, ...], c.F_d[i, ...])
        c.G[i, ...] = fibutils.func_G(c.m[i])

        if i > 1:
            c.s[i] = fibutils.func_s(c.r_abs[i], c.r_abs[i-1])
            c.ang[i] = fibutils.func_ang(c.e_r[i, ...], c.e_r[i-1, ...])
            c.cross_e[i] = fibutils.func_cross_e(c.e_r[i, ...], c.e_r[i-1, ...])
            c.cross_e_abs[i] = fibutils.func_cross_e_abs(c.cross_e[i])
            c.e_n[i, ...] = fibutils.func_e_n(c.cross_e[i], c.cross_e_abs[i])
            c.kappa[i, ...] = fibutils.func_kappa(c.ang[i], c.s[i], c.e_n[i, ...])
            c.kappadot[i, ...] = fibutils.func_kappadot(c.angdot[i, ...], c.angdot[i-1, ...], c.s[i])

            c.M_b[i, ...] = fibutils.func_M_b(c.kappa[i, ...])
            c.M_c[i, ...] = fibutils.func_M_c(c.kappadot[i, ...])
            c.M[i, ...] = fibutils.func_M(c.M_b[i, ...], c.M_c[i, ...])

        # The drag forces need to be defined for the coupled solver!
        # Otherwise, for only fiber dynamics, these are zero
        if config.params.coupled_solver:
            # Drag forces initialized to zero
            c.D_n[i, ...] = np.zeros(3)
            c.D_l[i, ...] = np.zeros(3)
            c.D[i, ...] = np.zeros(3)

    # Mass points n:
    c.Q[n, ...] = fibutils.func_Q(c.M[n, ...], np.zeros(3), c.r[n, ...], c.r_abs[n])

    if config.params.coupled_solver:
        if config.params.fib_fixed:
            c.gfun_old[6*n_plus-3 : 6*n_plus] = np.zeros(3)
        else:
            c.gfun_old[6*n_plus-3 : 6*n_plus] = fibutils.func_gfun_n(
                            c.m[i], c.G[n, ...], c.D[n, ...], c.F[n, ...], c.Q[n, ...])

    else:   # fiber dynamics solver alone
        if test in ["linstat", "lindyn"]:
            c.gfun_old[6*n_plus-3 : 6*n_plus] = np.zeros(3)

        elif test == "bend":
            c.gfun_old[6*n_plus-3] = fibutils.func_gfun_n(
                            c.m[i], c.G[n, ...], c.D[n, ...], c.F[n, ...], c.Q[n, ...])[0]
            c.gfun_old[6*n_plus-2] = 0
            c.gfun_old[6*n_plus-1] = 0

        else:
            c.gfun_old[6*n_plus-3 : 6*n_plus] = fibutils.func_gfun_n(
                            c.m[i], c.G[n, ...], c.D[n, ...], c.F[n, ...], c.Q[n, ...])

    c.xdotdot_old[n, ...] = np.copy(c.gfun_old[6*n_plus-3 : 6*n_plus])

    # loop in reverse order over mass points i = n-1, ..., 2, 1:
    for i in range(n-1, 0, -1):
        c.Q[i, ...] = fibutils.func_Q(c.M[i, ...], c.M[i+1, ...],
                                c.r[i, ...], c.r_abs[i])

        if i == config.params.fib_BC_index_F and test == "bend":
            c.gfun_old[3*n_plus + 3*i : 3*n_plus + 3*i+3] = fibutils.func_gfun(
                c.m[i], c.G[i, ...], c.D[i, ...],
                c.F[i, ...], c.Q[i, ...], c.F[i+1, ...], c.Q[i+1, ...]) +\
                np.array([0, 0, (1 / c.m[i]) * config.params.fib_BC_Fz])

        else:
            c.gfun_old[3*n_plus + 3*i : 3*n_plus + 3*i+3] = fibutils.func_gfun(
                c.m[i], c.G[i, ...], c.D[i, ...],
                c.F[i, ...], c.Q[i, ...], c.F[i+1, ...], c.Q[i+1, ...])

        c.xdotdot_old[i, ...] = np.copy(c.gfun_old[3*n_plus + 3*i : 3*n_plus + 3*i+3])

    # Mass points 0:
    if test == "linstat":
        c.gfun_old[3*n_plus : 3*n_plus + 3] = fibutils.func_gfun_0(
            c.m[0], c.G[0, ...], c.F[1, ...], c.Q[1, ...]) +\
            np.array([0, 0, (1 / c.m[0]) * config.params.fib_BC_Fz])

    elif test == "lindyn":
                                                        #c.gfun_old[3*n_plus : 3*n_plus + 3] = np.array([0, 0, fibutils.func_gfun_0(
                                                        #        c.m[0], c.G[0, ...], c.F[1, ...], c.Q[1, ...])[2] + (1 / c.m[0]) * Fz_dyn])
        c.gfun_old[3*n_plus : 3*n_plus + 3] = fibutils.func_gfun_0(
                c.m[0], c.G[0, ...], c.F[1, ...], c.Q[1, ...]) +\
                np.array([0, 0, (1 / c.m[0]) * Fz_dyn])

    elif test == "bend":
        c.gfun_old[3*n_plus : 3*n_plus + 3] = np.zeros(3)

    else:
        c.gfun_old[3*n_plus : 3*n_plus + 3] = fibutils.func_gfun_0(
            c.m[0], c.G[0, ...], c.F[1, ...], c.Q[1, ...])

    c.xdotdot_old[0, ...] = np.copy(c.gfun_old[3*n_plus : 3*n_plus + 3])

    # NOTE: ndarray to list (A of type ndarray) -> B = np.ndarray.tolist(A)
    # NOTE: list to ndarray (B of type list) -> A = np.array(B)


if __name__ == "__main__":

    ###----------------------------- PARAMETERS -----------------------------###

    # Fiber dynamics solver arguments:
    config.update(
        {'coupled_solver': False,
         'fib_n': 8,
         'fib_rho': 7800.,
         'fib_d': 6e-3,
         'fib_L': 0.2,
         'fib_E': 210.0e9,
         'fib_eta': 500.0e3,
         'g': -9.80665,
         'fib_m0': 1,
         'fib_n_threads': 1,
         'fib_L0': 1.,
         'fib_n_plus': 9,
         'fib_A': 1.,
         'fib_I': 1.,
         'fib_m_other': 1.,
         'fib_ratioLd': 1.,
         'nu': 0.005428,
         'dt': 0.001,
         'T': 0.05,
         'L': [2.*np.pi, 2.*np.pi, 2.*np.pi],
        }, "triplyperiodic"
    )

    # Additional arguments of the Fiber Dynamics Solver:
    config.triplyperiodic.add_argument("--fib_BC_Fz", type=float,
        default=-10000., help="[N]")
    config.triplyperiodic.add_argument("--fib_BC_index_F", type=int,
        default=0, help="Index of the mass point to which the force is applied.")
    config.triplyperiodic.add_argument("--fib_test", type=str,
        default='notest', choices=('linstat', 'lindyn', 'bend', 'notest'),
        help="Which test case for validation.")
    config.triplyperiodic.add_argument("--fib_w_forced_ratio", type=float,
        default=0., help="Forcing frequency ratio: ratio = w_forced / w_0.")
    config.triplyperiodic.add_argument("--fib_k", type=float,
        default=1., help="Stiffness of the fiber")
    config.triplyperiodic.add_argument("--fib_m", type=float,
        default=1., help="Total mass of the fiber")
    config.triplyperiodic.add_argument("--fib_w_0", type=float,
        default=1., help="Natural frequency of the harmonic oscillator [rad/s]")
    config.triplyperiodic.add_argument("--fib_w_forced", type=float,
        default=1., help="Forcing frequency of the harmonic oscillator [rad/s]")

    mesh="triplyperiodic"
    parse_args = None
    assert parse_args is None or isinstance(parse_args, list)
    args = getattr(getattr(config, mesh), 'parse_args')(parse_args)
    config.params.update(vars(args))
    # Here, the arguments given to the program update the parameters

    ### PARAMETERS OF THE VALIDATION TESTS ###
    #config.params.fib_test = "lindyn" # linstat lindyn bend

    # Define manually the parameters of the validation tests. Can also set the
    # parameters as arguments when calling the module (set right above):
    #config.params.fib_test = "lindyn" # options: "linstat", "lindyn", "bend",
    #"notest" -> set as argument

    if config.params.fib_test == "linstat":
        print("\n---------- LINEAR STATIC VALIDATION TEST CASE ----------\n")
        config.params.fib_n = 1     # n = 1, 2, 4, 8, 16
        config.params.fib_eta = 500.0e3
        config.params.dt = 0.00001
        #config.params.dt = 0.0001
        config.params.T = 0.05
        config.params.fib_BC_Fz = -10000
    elif config.params.fib_test == "lindyn":
        print("\n---------- LINEAR DYNAMIC VALIDATION TEST CASE ----------\n")
        config.params.fib_n = 4
        #config.params.fib_eta = 500.0e3     # eta = 500.0e3, 8500.0e3, 76500.0e3
        config.params.dt = 0.0000001
        #config.params.dt = 0.0000002
        config.params.T = 0.1
        config.params.fib_BC_Fz = -500.
        #config.params.fib_w_forced_ratio = 0.5      # w_forced = 0., 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.
    elif config.params.fib_test == "bend":
        print("\n---------- BENDING VALIDATION TEST CASE ----------\n")
        config.params.fib_n = 2    # n = 2, 4, 8, 16
        config.params.fib_eta = 8500.0e3
        config.params.dt = 0.00001
        config.params.T = 0.05
        config.params.fib_BC_Fz = -200
        config.params.fib_BC_index_F = config.params.fib_n // 2

    # Additional parameters:
    config.params.nu = config.params.fib_L / config.params.fib_n
    config.params.fib_L0 = config.params.fib_L / config.params.fib_n
    config.params.fib_n_plus = config.params.fib_n + 1 # number of mass points including the spider
    config.params.fib_A = np.pi * config.params.fib_d**2 / 4
    config.params.fib_I = np.pi * config.params.fib_d**4 / 64
    config.params.fib_m_other = config.params.fib_rho * config.params.fib_L *\
                                    config.params.fib_A / config.params.fib_n
    config.params.fib_ratioLd = config.params.fib_L / config.params.fib_d
    config.params.fib_BC_index_fix = config.params.fib_n

    # Additional parameters for the validation tests:
    if config.params.fib_test == "lindyn":
        config.params.fib_k = config.params.fib_E *\
                                config.params.fib_A / config.params.fib_L
        config.params.fib_m = config.params.fib_m0 + config.params.fib_rho *\
                                config.params.fib_A * config.params.fib_L / 3
        config.params.fib_w_0 = np.sqrt(config.params.fib_k / config.params.fib_m)
        config.params.fib_w_forced = config.params.fib_w_forced_ratio *\
                                        config.params.fib_w_0

    ###--------------------- CONTEXT AND INITIALIZATION ---------------------###

    # Get the context to solve the fibre dynamics:
    fib_context = get_fibcontext()
    initialize_fibre(fib_context)

    # Check the values of all the parameters defined:
    fibutils.check_parameters()

    ###------------------------- ITERATIVE PROCESS --------------------------###

    #fibutils.check_variables(fib_context)

    # Set timer:
    timer = Timer()

    # Set further parameters to call the functions:
    dt_in = config.params.dt
    n = config.params.fib_n
    n_plus = config.params.fib_n_plus
    test = config.params.fib_test
    if test != "lindyn":
        Fz_dyn = 0

    # Time loop:
    # Initially: config.params.t = 0.0 -> see spectralDNS.config
    while config.params.t + config.params.dt <= config.params.T+1e-12:

        dt_took = config.params.dt
        config.params.t += dt_took
        config.params.tstep += 1

        # Set the value of the sinusoidal force for the linear dynamic test case
        # depending on the forcing frequency:
        if test == "lindyn":
            if config.params.fib_w_forced_ratio == 0.0:
                Fz_dyn = config.params.fib_BC_Fz
            else:
                Fz_dyn = config.params.fib_BC_Fz *\
                    np.sin(config.params.fib_w_forced * config.params.t)

        # Call the iterative solver to get the solution at the next time step:
        fibutils.advance_fibre(fib_context, n, n_plus, dt_took, test, Fz_dyn)

        # Print to track the evolution of the simulation:
        if config.params.tstep % 100 == 0:
            print("converged ", config.params.tstep)

        # Update the variables for the new time step:
        fibutils.update_fibre(fib_context, n, n_plus, test, Fz_dyn)

        #fibutils.check_variables(fib_context) # -> can check variables at any time

        # I/O:
        fibutils.writetofile(fib_context, n_plus, test, Fz_dyn)

        #if config.params.tstep == 4000:
        #    fibutils.check_variables(fib_context)

        timer()

        if fibutils.end_of_tstep(fib_context):
            break

    config.params.dt = dt_in

    ###--------------------------- POSTPROCESSING ---------------------------###

    if config.params.fib_test == "linstat":
        fibutils.visualize_linstat(fib_context)

    elif config.params.fib_test == "lindyn":
        fibutils.visualize_lindyn(fib_context)

    elif config.params.fib_test == "bend":
        fibutils.visualize_bend(fib_context)

    # Print times:
    timer.final(config.params.verbose)

# SCIPY:
#
# https://docs.scipy.org/doc/scipy/reference/optimize.html

# CN integration method:
#
# http://www.claudiobellei.com/2016/11/10/crank-nicolson/

# TRICUBIC interpolation:
#
# file:///C:/Users/jonny/Desktop/ETH%20Zurich/MSc%20Mechanical%20Engineering/Spring%20Semester%202020/Semester%20Project/References/Tricubic_interpolation_2005.pdf
# https://github.com/danielguterding/pytricubic/blob/master/example.py
