import numpy as np
from spectralDNS import config
from scipy.optimize import root
from fibredyn.drag_coeffs import drag_coeff_normal, drag_coeff_long
import csv
import matplotlib.pyplot as plt
from mpi4py import MPI

def distribute_fibres(nprocs, rank):
    """returns for each rank the amount of fibre the rank has to process"""
    div=int(config.params.fib_amt/nprocs)
    rem=config.params.fib_amt%nprocs
    if rem>rank:
        return int(div+1)
    else:
        return int(div)

def fib_num(rank, nprocs, fib_per_prc):
    """returns number of first fibre of rank"""
    disc=config.params.fib_amt%nprocs
    if disc > rank:
        return rank*fib_per_prc
    else:
        return rank*fib_per_prc + disc

def allocate_fib(fib_id, nprocs):
    """allocates a fibre to a rank, based on its number"""
    for i in range(nprocs):
        fib_per_prc=distribute_fibres(nprocs, i)
        current_prc=fib_num(i, nprocs, fib_per_prc)
        if fib_id < (current_prc + fib_per_prc):
            return i
    assert False


def fib_orientation_init(context, start, axis, vel_ini = 0., axis_vel = 3):
    c = context
    n = config.params.fib_n
    n_plus = config.params.fib_n_plus

    ## Initialize positions and velocities (add more possibilities):
    # The 3D domain is 2*pi x 2*pi x 2*pi
    start_pos_x1 = start[0]
    start_pos_x2 = start[1]
    start_pos_x3 = start[2]

    for i in range(n_plus):
        # Initialize the positions of the fiber:
        if axis == 1:
            c.x_old[i, 0] = start_pos_x1 + i*config.params.fib_L0
            c.x_old[i, 1] = start_pos_x2
            c.x_old[i, 2] = start_pos_x3
        elif axis == 2:
            c.x_old[i, 0] = start_pos_x1
            c.x_old[i, 1] = start_pos_x2 + i*config.params.fib_L0
            c.x_old[i, 2] = start_pos_x3
        elif axis == 3:
            c.x_old[i, 0] = start_pos_x1
            c.x_old[i, 1] = start_pos_x2
            c.x_old[i, 2] = start_pos_x3 + i*config.params.fib_L0
        # Initialize the velocities of the fiber:
        if vel_ini != 0.:
            if axis_vel == 1:
                c.xdot_old[i, 0] = vel_ini
                c.xdot_old[i, 1] = 0
                c.xdot_old[i, 2] = 0
            elif axis_vel == 2:
                c.xdot_old[i, 0] = 0
                c.xdot_old[i, 1] = vel_ini
                c.xdot_old[i, 2] = 0
            elif axis_vel == 3:
                c.xdot_old[i, 0] = 0
                c.xdot_old[i, 1] = 0
                c.xdot_old[i, 2] = vel_ini
        else:
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

###-------------------------- FIBER DYNAMICS FUNCTIONS -------------------------###

# _iplus -> index i-1
# _i -> index i
# _imin -> index i+1

def func_r(pos_i, pos_imin):
    assert pos_i.shape[0]==pos_imin.shape[0]
    assert len (pos_i.shape)==1
    assert len (pos_imin.shape)==1
    res=np.zeros(pos_i.size)
    #check for segments that cross the boundary
    for i in range (pos_i.size):
        if abs(pos_i[i]-pos_imin[i])>2*config.params.fib_L0:
            if pos_i[i]<pos_imin[i]:
                res[i]=pos_i[i] - pos_imin[i] + config.params.L[i]
            else:
                res[i]=pos_i[i] - pos_imin[i] - config.params.L[i]
        else:
            res[i]=pos_i[i] - pos_imin[i]

    return res

def func_r_abs(r):
    return np.linalg.norm(r)

def func_e_r(r, r_abs):
    return r / r_abs

def func_rdot(xdot_i, xdot_imin):
    return xdot_i - xdot_imin

def func_angdot(e_r, rdot, r_abs):
    return np.cross(e_r, rdot) / r_abs

def func_eps(r_abs):
    return (r_abs - config.params.fib_L0) / config.params.fib_L0

def func_epsdot(e_r, rdot):
    return np.dot(rdot, e_r) / config.params.fib_L0

def func_F_k(eps, e_r):
    return - config.params.fib_E * config.params.fib_A * eps * e_r

def func_F_d(epsdot, e_r):
    return - config.params.fib_eta * config.params.fib_A * epsdot * e_r

def func_F(F_k, F_d):
    return F_k + F_d

def func_G(m):
    if config.params.validate==False:
        return np.array([0, 0, m *(1 - config.params.rho / config.params.fib_rho) * config.params.g])
    else:
        return np.array([0, 0, m * config.params.g])
def func_s(r_abs_i, r_abs_imin):
    return 0.5 * (r_abs_imin + r_abs_i)

def func_ang(e_r_i, e_r_imin):
    dote = np.dot(e_r_imin, e_r_i)
    # Avoid |dote|>1.0 due to rounding errors calculating the e_r's and dot product:
    if dote < -1.:
        dote = -1.
    elif dote > 1.:
        dote = 1.
    #assert dote >= -1 and dote <= 1.
    return np.arccos(dote)

def func_cross_e(e_r_i, e_r_imin):
    return np.cross(e_r_imin, e_r_i)

def func_cross_e_abs(cross):
    return np.linalg.norm(cross)

def func_e_n(cross, cross_abs):
    if cross_abs == 0:
        return np.array([0, 0, 0])
    else:
        return cross / cross_abs

def func_kappa(ang, s, e_n):
    return (ang / s) * e_n

def func_kappadot(angdot_i, angdot_imin, s):
    return (angdot_i - angdot_imin) / s

def func_M_b(kappa):
    return - config.params.fib_E * config.params.fib_I * kappa

def func_M_c(kappadot):
    return - config.params.fib_eta * config.params.fib_I * kappadot

def func_M(M_b, M_c):
    return M_b + M_c

def func_D_n(e_r, v_rel):
    v_rel_abs = np.linalg.norm(v_rel)
    if v_rel_abs == 0.:
        return np.zeros(3)
    else:
        coef = np.dot(e_r, v_rel)   # e_r is a unit vector, don't divide by |e_r|^2
        if coef == 0.:
            v_rel_n = v_rel
            alpha == np.pi / 2
            e_perp = v_rel_n / np.linalg.norm(v_rel_n)
        elif coef == v_rel_abs:
            v_rel_n = np.zeros(3)
            alpha == 0.
            e_perp = np.zeros(3)
        else:
            v_rel_n = v_rel - coef * e_r
            v_rel_l = coef * e_r
            acos_in = np.dot(v_rel, v_rel_l) / (v_rel_abs * np.linalg.norm(v_rel_l))
            if acos_in < -1.:
                acos_in = -1.
            elif acos_in > 1.:
                acos_in = 1.
            alpha = np.arccos(acos_in)
            e_perp = v_rel_n / np.linalg.norm(v_rel_n)
        D_n_abs = 0.5 * config.params.rho * config.params.fib_d * v_rel_abs**2 * \
                    drag_coeff_normal(v_rel_abs, alpha)
        return D_n_abs * config.params.fib_L0 * e_perp

def func_D_l(e_r, v_rel):
    v_rel_abs = np.linalg.norm(v_rel)
    if v_rel_abs == 0.:
        return np.zeros(3)
    else:
        coef = np.dot(e_r, v_rel)   # e_r is a unit vector, don't divide by |e_r|^2
        if coef == 0.:
            v_rel_l = np.zeros(3)
            alpha == np.pi / 2
            e_parall = np.zeros(3)
        elif coef == v_rel_abs:
            v_rel_l = v_rel
            alpha == 0.
            e_parall = v_rel_l / np.linalg.norm(v_rel_l)
        else:
            v_rel_n = v_rel - coef * e_r
            v_rel_l = coef * e_r
            acos_in = np.dot(v_rel, v_rel_l) / (v_rel_abs * np.linalg.norm(v_rel_l))
            if acos_in < -1.:
                acos_in = -1.
            elif acos_in > 1.:
                acos_in = 1.
            alpha = np.arccos(acos_in)
            e_parall = v_rel_l / np.linalg.norm(v_rel_l)
        D_l_abs = 0.5 * config.params.rho * config.params.fib_d * v_rel_abs**2 * \
                    drag_coeff_long(v_rel_abs, alpha, config.params.fib_ratioLd)
        return D_l_abs * config.params.fib_L0 * e_parall

def func_D(D_n, D_l):
    return D_n + D_l

def func_Q(M_i, M_iplus, r, r_abs):
    return np.cross(M_i - M_iplus, r) / r_abs**2

def func_gfun_n(m, G, D, F, Q):
    if config.params.validate==False:
        return (1 / m ) * (G + D + F + Q)
    else:
        return (1 /( m ) * (G + D))

def func_gfun(m, G, D, F_i, Q_i, F_iplus, Q_iplus):
    if config.params.validate==False:
        return (1 / m ) * (G + D + F_i + Q_i - F_iplus - Q_iplus)
    else:
        return (1 / (m) * (G + D))

def func_gfun_0(m, G, F, Q):
    if config.params.validate==False:
        return (1 / m ) * (G - config.params.fib_n_threads * (F + Q))
    else:
        return (1 / (m) * (G))

def fun_CrankNicolson(z, c, n, n_plus, dt, test='notest', Fz_dyn=0):
    """
    Implementation of the Crank-Nicolson method to the system of nonlinear
    coupled first-order ODE's as cast in the report, from the system of
    second-order ODE's
    """

    # c = context
    c._G[0] =  func_G(c.m[0])
    for i in range(1, n_plus):
        c._r[i] = func_r(np.array([z[3*i], z[3*i+1], z[3*i+2]]), np.array([z[3*i-3], z[3*i-2], z[3*i-1]]))
        c._r_abs[i] = func_r_abs(c._r[i])
        c._e_r[i] = func_e_r(c._r[i], c._r_abs[i])
        c._rdot[i] = func_rdot(np.array([z[3*n_plus+3*i], z[3*n_plus+3*i+1], z[3*n_plus+3*i+2]]), np.array([z[3*n_plus+3*i-3], z[3*n_plus+3*i-2], z[3*n_plus+3*i-1]]))
        c._angdot[i] = func_angdot(c._e_r[i], c._rdot[i], c._r_abs[i])
        c._eps[i] = func_eps(c._r_abs[i])
        c._epsdot[i] = func_epsdot(c._e_r[i], c._rdot[i])

        c._F_k[i] = func_F_k(c._eps[i], c._e_r[i])
        c._F_d[i] = func_F_d(c._epsdot[i], c._e_r[i])
        c._F[i] = func_F(c._F_k[i], c._F_d[i])
        c._G[i] =  func_G(c.m[i])

        if i > 1:
            c._s[i] = func_s(c._r_abs[i], c._r_abs[i-1])
            c._ang[i] = func_ang(c._e_r[i], c._e_r[i-1])
            c._cross_e[i] = func_cross_e(c._e_r[i], c._e_r[i-1])
            c._cross_e_abs[i] = func_cross_e_abs(c._cross_e[i])
            c._e_n[i] = func_e_n(c._cross_e[i], c._cross_e_abs[i])
            c._kappa[i] = func_kappa(c._ang[i], c._s[i], c._e_n[i])
            c._kappadot[i] = func_kappadot(c._angdot[i], c._angdot[i-1], c._s[i])

            c._M_b[i] = func_M_b(c._kappa[i])
            c._M_c[i] = func_M_c(c._kappadot[i])
            c._M[i] = func_M(c._M_b[i], c._M_c[i])

        if config.params.coupled_solver:
            c._D_n[i] = func_D_n(c._e_r[i], c.v_rel[i])
            c._D_l[i] = func_D_l(c._e_r[i], c.v_rel[i])
            c._D[i] = func_D(c._D_n[i], c._D_l[i])
    # Mass points n:
    c._Q[n] = func_Q(c._M[n], np.zeros(3), c._r[n], c._r_abs[n])

    if config.params.coupled_solver:
        if config.params.fib_fixed:
            c._gfun[6*n_plus-3] = 0
            c._gfun[6*n_plus-2] = 0
            c._gfun[6*n_plus-1] = 0
        else:
            c._gfun[6*n_plus-3] = func_gfun_n(c.m[i], c._G[n], c._D[n], c._F[n], c._Q[n])[0]
            c._gfun[6*n_plus-2] = func_gfun_n(c.m[i], c._G[n], c._D[n], c._F[n], c._Q[n])[1]
            c._gfun[6*n_plus-1] = func_gfun_n(c.m[i], c._G[n], c._D[n], c._F[n], c._Q[n])[2]

    else:   # fiber dynamics solver alone
        if test in ["linstat", "lindyn"]:
            c._gfun[6*n_plus-3] = 0
            c._gfun[6*n_plus-2] = 0
            c._gfun[6*n_plus-1] = 0
        elif test == "bend":
            c._gfun[6*n_plus-3] = func_gfun_n(c.m[i], c._G[n], c._D[n], c._F[n], c._Q[n])[0]
            c._gfun[6*n_plus-2] = 0
            c._gfun[6*n_plus-1] = 0
        else:
            c._gfun[6*n_plus-3] = func_gfun_n(c.m[i], c._G[n], c._D[n], c._F[n], c._Q[n])[0]
            c._gfun[6*n_plus-2] = func_gfun_n(c.m[i], c._G[n], c._D[n], c._F[n], c._Q[n])[1]
            c._gfun[6*n_plus-1] = func_gfun_n(c.m[i], c._G[n], c._D[n], c._F[n], c._Q[n])[2]

    c._xdotdot[n] = np.array([c._gfun[6*n_plus-3], c._gfun[6*n_plus-2], c._gfun[6*n_plus-1]])

    # loop in reverse order over mass points i = n-1, ..., 2, 1:
    for i in range(n-1, 0, -1):
        c._Q[i] = func_Q(c._M[i], c._M[i+1], c._r[i], c._r_abs[i])
        c._gfun[3*n_plus + 3*i] = func_gfun(
                c.m[i], c._G[i], c._D[i], c._F[i], c._Q[i], c._F[i+1], c._Q[i+1])[0]
        c._gfun[3*n_plus + 3*i+1] = func_gfun(
                c.m[i], c._G[i], c._D[i], c._F[i], c._Q[i], c._F[i+1], c._Q[i+1])[1]
        c._gfun[3*n_plus + 3*i+2] = func_gfun(
                c.m[i], c._G[i], c._D[i], c._F[i], c._Q[i], c._F[i+1], c._Q[i+1])[2]

    c._xdotdot[n] = np.array([c._gfun[6*n_plus-3], c._gfun[6*n_plus-2], c._gfun[6*n_plus-1]])

    # Mass points 0:
    c._gfun[3*n_plus] = func_gfun_0(c.m[0], c._G[0], c._F[1], c._Q[1])[0]
    c._gfun[3*n_plus+1] = func_gfun_0(c.m[0], c._G[0], c._F[1], c._Q[1])[1]
    c._gfun[3*n_plus+2] = func_gfun_0(c.m[0], c._G[0], c._F[1], c._Q[1])[2]
    c._xdotdot[0] = np.array([c._gfun[3*n_plus], c._gfun[3*n_plus+1], c._gfun[3*n_plus+2]])

    for i in range(3*n_plus):
        c._gfun[i] = z[3*n_plus+i]


    # Crank - Nicolson method:
    for j in range(6*n_plus):
        c._h[j] = c.z_old[j] - z[j] + (dt / 2) * (c.gfun_old[j] + c._gfun[j])
    return c._h


def advance_fibre(c, n, n_plus, dt, test='notest', Fz_dyn=0):
    """
    Iterative solver to solve the system of nonlinear coupled algebraic
    equations that result from the time integration by the implicit CN method
    """
    # c = context
    # Use previous z_old as initial guess
    solution = root(fun_CrankNicolson, c.z_old, (c, n, n_plus, dt, test, Fz_dyn),
                    method='hybr', options={'xtol': 1.49012e-08, 'maxfev': 1500}) # a bit faster
    #solution = root(fun_CrankNicolson, c.z_old, (c, n, n_plus, dt, test, Fz_dyn),
    #                method='lm', options={'xtol': 1.49012e-08, 'maxiter': 1000}) # a bit faster

    #                   solution = root(fun_CrankNicolson, c.z_old, (c, n, n_plus, dt, test, Fz_dyn), method='lm')

    #                   solution = root(fun_CrankNicolson, c.z_old, (c, n, n_plus, dt, test, Fz_dyn), method='lm',
    #    options={
    #    'xtol': 1.49012e-15, 'ftol': 1.49012e-15, 'gtol': 0.0,
    #    'maxiter': 0, 'eps': 0.0, 'factor': 100, 'diag': None
    #    })
    print(solution.success, solution.message)
    c.z_new = np.copy(solution.x)   #type(solution.x) -> np.ndarray

    #count if convergence was not reached
    if solution.success==False:
        c.convergence+=1

    #relocate joints outside of domain, and acount for the potential energy
    for i in range(n_plus):
        for j in range(3):
            if c.z_new[3*i+j]>=config.params.L[j]:
                c.z_new[3*i+j]-=config.params.L[j]
                if j==2:
                    c.periodic += 1
            elif c.z_new[3*i+j]<0:
                c.z_new[3*i+j]+=config.params.L[j]
                if j==2:
                    c.periodic -= 1
    if MPI.COMM_WORLD.Get_rank()==0:
        print("\nNumber of objective function evauations from rank 0: ", solution.nfev)
    c.nfev.append(solution.nfev)

    ### See: https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/_root.py#L24-L203


def update_fibre(c, n, n_plus, test='notest', Fz_dyn=0):
    """
    Update the variables to get gfun_new and z_new, used in the Following
    time step as gfun_old and z_old
    """
    print("update")
    for i in range(n_plus):
        c.x_new[i, 0] = np.copy(c.z_new[3*i])
        c.x_new[i, 1] = np.copy(c.z_new[3*i+1])
        c.x_new[i, 2] = np.copy(c.z_new[3*i+2])

        c.xdot_new[i, 0] = np.copy(c.z_new[n_plus*3+3*i])
        c.xdot_new[i, 1] = np.copy(c.z_new[n_plus*3+3*i+1])
        c.xdot_new[i, 2] = np.copy(c.z_new[n_plus*3+3*i+2])

        c.gfun_new[3*i] = np.copy(c.xdot_new[i, 0])
        c.gfun_new[3*i+1] = np.copy(c.xdot_new[i, 1])
        c.gfun_new[3*i+2] = np.copy(c.xdot_new[i, 2])

        c.x_old[i, 0] = np.copy(c.x_new[i, 0])
        c.x_old[i, 1] = np.copy(c.x_new[i, 1])
        c.x_old[i, 2] = np.copy(c.x_new[i, 2])

        c.xdot_last[i,0] = np.copy(c.xdot_old[i,0])
        c.xdot_last[i,1] = np.copy(c.xdot_old[i,1])
        c.xdot_last[i,2] = np.copy(c.xdot_old[i,2])

        c.xdot_old[i, 0] = np.copy(c.xdot_new[i, 0])
        c.xdot_old[i, 1] = np.copy(c.xdot_new[i, 1])
        c.xdot_old[i, 2] = np.copy(c.xdot_new[i, 2])

        c.z_old[3*i] = np.copy(c.z_new[3*i])
        c.z_old[3*i+1] = np.copy(c.z_new[3*i+1])
        c.z_old[3*i+2] = np.copy(c.z_new[3*i+2])
        c.z_old[n_plus*3+3*i] = np.copy(c.z_new[n_plus*3+3*i])
        c.z_old[n_plus*3+3*i+1] = np.copy(c.z_new[n_plus*3+3*i+1])
        c.z_old[n_plus*3+3*i+2] = np.copy(c.z_new[n_plus*3+3*i+2])

        c.gfun_old[3*i] = np.copy(c.gfun_new[3*i])
        c.gfun_old[3*i+1] = np.copy(c.gfun_new[3*i+1])
        c.gfun_old[3*i+2] = np.copy(c.gfun_new[3*i+2])
    c.G[0, ...] = func_G(c.m[0])
    if config.params.coupled_solver:
        c.D_last[:]=c.D[:]
    for i in range(1, n_plus):
        c.r[i, ...] = func_r(c.x_new[i, ...], c.x_new[i-1, ...])
        c.r_abs[i] = func_r_abs(c.r[i, ...])
        c.e_r[i, ...] = func_e_r(c.r[i, ...], c.r_abs[i])
        c.rdot[i, ...] = func_rdot(c.xdot_new[i, ...], c.xdot_new[i-1, ...])
        c.angdot[i, ...] = func_angdot(c.e_r[i, ...], c.rdot[i, ...], c.r_abs[i])
        c.eps[i] = func_eps(c.r_abs[i])
        c.epsdot[i] = func_epsdot(c.e_r[i, ...], c.rdot[i, ...])

        c.F_k[i, ...] = func_F_k(c.eps[i], c.e_r[i, ...])
        c.F_d[i, ...] = func_F_d(c.epsdot[i], c.e_r[i, ...])
        c.F[i, ...] = func_F(c.F_k[i, ...], c.F_d[i, ...])
        c.G[i, ...] = func_G(c.m[i])

        if i > 1:
            c.s[i] = func_s(c.r_abs[i], c.r_abs[i-1])
            c.ang[i] = func_ang(c.e_r[i, ...], c.e_r[i-1, ...])
            c.cross_e[i] = func_cross_e(c.e_r[i, ...], c.e_r[i-1, ...])
            c.cross_e_abs[i] = func_cross_e_abs(c.cross_e[i])
            c.e_n[i, ...] = func_e_n(c.cross_e[i], c.cross_e_abs[i])
            c.kappa[i, ...] = func_kappa(c.ang[i], c.s[i], c.e_n[i, ...])
            c.kappadot[i, ...] = func_kappadot(c.angdot[i, ...], c.angdot[i-1, ...], c.s[i])

            c.M_b[i, ...] = func_M_b(c.kappa[i, ...])
            c.M_c[i, ...] = func_M_c(c.kappadot[i, ...])
            c.M[i, ...] = func_M(c.M_b[i, ...], c.M_c[i, ...])

        # The drag forces need to be defined for the coupled solver!
        # Otherwise, for only fiber dynamics, these are zero
        if config.params.coupled_solver:
            c.D_n[i, ...] = func_D_n(c.e_r[i, ...], c.v_rel[i, ...])
            c.D_l[i, ...] = func_D_l(c.e_r[i, ...], c.v_rel[i, ...])
            c.D[i, ...] = func_D(c.D_n[i, ...], c.D_l[i, ...])
    # Mass points n:
    c.Q[n, ...] = func_Q(c.M[n, ...], np.zeros(3), c.r[n, ...], c.r_abs[n])

    if config.params.coupled_solver:
        if config.params.fib_fixed:
            c.gfun_new[6*n_plus-3 : 6*n_plus] = np.zeros(3)
        else:
            c.gfun_new[6*n_plus-3 : 6*n_plus] = func_gfun_n(
                            c.m[i], c.G[n, ...], c.D[n, ...], c.F[n, ...], c.Q[n, ...])

    else:   # fiber dynamics solver alone
        if test in ["linstat", "lindyn"]:
            c.gfun_new[6*n_plus-3 : 6*n_plus] = np.zeros(3)

        elif test == "bend":
            c.gfun_new[6*n_plus-3] = func_gfun_n(
                            c.m[i], c.G[n, ...], c.D[n, ...], c.F[n, ...], c.Q[n, ...])[0]
            c.gfun_new[6*n_plus-2] = 0
            c.gfun_new[6*n_plus-1] = 0

        else:
            c.gfun_new[6*n_plus-3 : 6*n_plus] = func_gfun_n(
                            c.m[i], c.G[n, ...], c.D[n, ...], c.F[n, ...], c.Q[n, ...])

    c.xdotdot_new[n, ...] = np.copy(c.gfun_new[6*n_plus-3 : 6*n_plus])

    # loop in reverse order over mass points i = n-1, ..., 2, 1:
    for i in range(n-1, 0, -1):
        c.Q[i, ...] = func_Q(c.M[i, ...], c.M[i+1, ...],
                                c.r[i, ...], c.r_abs[i])

        if i == config.params.fib_BC_index_F and test == "bend":
            c.gfun_new[3*n_plus + 3*i : 3*n_plus + 3*i+3] = func_gfun(
                c.m[i], c.G[i, ...], c.D[i, ...],
                c.F[i, ...], c.Q[i, ...], c.F[i+1, ...], c.Q[i+1, ...]) +\
                np.array([0, 0, (1 / c.m[i]) * config.params.fib_BC_Fz])

        else:
            c.gfun_new[3*n_plus + 3*i : 3*n_plus + 3*i+3] = func_gfun(
                c.m[i], c.G[i, ...], c.D[i, ...],
                c.F[i, ...], c.Q[i, ...], c.F[i+1, ...], c.Q[i+1, ...])

        c.xdotdot_new[i, ...] = np.copy(c.gfun_new[3*n_plus + 3*i : 3*n_plus + 3*i+3])

    # Mass points 0:
    if test == "linstat":
        c.gfun_new[3*n_plus : 3*n_plus + 3] = func_gfun_0(
            c.m[0], c.G[0, ...], c.F[1, ...], c.Q[1, ...]) +\
            np.array([0, 0, (1 / c.m[0]) * config.params.fib_BC_Fz])

    elif test == "lindyn":
        c.gfun_new[3*n_plus : 3*n_plus + 3] = func_gfun_0(
                c.m[0], c.G[0, ...], c.F[1, ...], c.Q[1, ...]) +\
                np.array([0, 0, (1 / c.m[0]) * Fz_dyn])

    elif test == "bend":
        c.gfun_new[3*n_plus : 3*n_plus + 3] = np.zeros(3)

    else:
        c.gfun_new[3*n_plus : 3*n_plus + 3] = func_gfun_0(
            c.m[0], c.G[0, ...], c.F[1, ...], c.Q[1, ...])

    c.xdotdot_new[0, ...] = np.copy(c.gfun_new[3*n_plus : 3*n_plus + 3])



    # Update gfun for further use by the CN method:
    for i in range(n_plus):
        c.gfun_old[n_plus*3+3*i] = np.copy(c.gfun_new[n_plus*3+3*i])
        c.gfun_old[n_plus*3+3*i+1] = np.copy(c.gfun_new[n_plus*3+3*i+1])
        c.gfun_old[n_plus*3+3*i+2] = np.copy(c.gfun_new[n_plus*3+3*i+2])
        c.xdotdot_old[i, ...] = np.copy(c.xdotdot_new[i, ...])


def end_of_tstep(context):
    # Make sure that the last step hits T exactly.
    # Used by adaptive solvers
    if abs(config.params.t - config.params.T) < 1e-12:
        return True

    if (abs(config.params.t + config.params.dt - config.params.T) < 1e-12 or
            config.params.t + config.params.dt >= config.params.T + 1e-12):
        config.params.dt = config.params.T - config.params.t

    return False

def writetofile(c, n_plus, test='notest', Fz_dyn=0):
    """ I/O routine. Mainly used in the validation tests """

    if test == "linstat":

        stringfile = "./csv_files/visualization_lin_" + str(config.params.fib_n) + ".csv"

        if config.params.tstep == 1:
            with open(stringfile, mode='w', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow([c.x_new[0, 2], config.params.t, config.params.fib_BC_Fz])

        else:
            with open(stringfile, mode='a+', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow([c.x_new[0, 2], config.params.t, config.params.fib_BC_Fz])

    elif test == "lindyn":

        stringfile = "./csv_files/visualization_dyn_" + str(int(config.params.fib_eta/1000)) + "_" + str(config.params.fib_w_forced_ratio) + ".csv"

        if config.params.tstep == 1:
            with open(stringfile, mode='w', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow([c.x_new[0, 2], config.params.t, Fz_dyn, c.x_new[0, 1], c.x_new[0, 0]])

        else:
            with open(stringfile, mode='a+', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow([c.x_new[0, 2], config.params.t, Fz_dyn, c.x_new[0, 1], c.x_new[0, 0]])

    elif test == "bend":

        stringfile = "./csv_files/visualization_bend_" + str(config.params.fib_n) + ".csv"

        if config.params.tstep == 1:
            with open(stringfile, mode='w', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow([c.x_new[n_plus//2, 2], config.params.t, config.params.fib_BC_Fz])

        else:
            with open(stringfile, mode='a+', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow([c.x_new[n_plus//2, 2], config.params.t, config.params.fib_BC_Fz])

def calc_velocity(vector):
    assert (vector.shape!=3 or vector.shape!=2)
    return np.linalg.norm(vector)

def compute_fib_energy(context):
    c=context
    E_kin_new=0
    E_pot_new=0
    W_f_new=0
    for i in range(c.x_new.shape[0]):
        v=calc_velocity(c.xdot_old[i,:])
        v_last=calc_velocity(c.xdot_last[i,:])
        m=c.m[i]
        assert (v >= 0 and m >= 0)
        E_kin_new += 0.5*m*v**2
        E_pot_new -= m*config.params.g*c.x_old[i,2]
        W_f_new += (np.dot(c.xdot_old[i,:],c.D[i,:])+np.dot(c.xdot_last[i,:],c.D_last[i,:]))*0.5*config.params.dt
        #ensures refreshment of velocity and mass computation
        v=-1
        m=-1

    E_pot_new -= c.periodic * config.params.L[2] * c.m[-1] * config.params.g
    c.E_kin.append(E_kin_new)
    c.E_pot.append(E_pot_new)
    c.W_f.append(W_f_new)
    c.E_is.append(E_kin_new + E_pot_new)
    if len(c.E_pr)==0:
        c.E_pr.append(c.E_is[0])
    else:
        c.E_pr.append(c.E_pr[-1]+W_f_new)
    c.delta.append(c.E_is[-1]-c.E_pr[-1])

def visualize_linstat(context):
    x = []
    time = []
    stringfile = "./csv_files/visualization_lin_" + str(config.params.fib_n) + ".csv"
    with open(stringfile, newline='') as csvfile:
        visualreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in visualreader:
            x.append(abs(float(row[0])))
            time.append(float(row[1]))

    k = config.params.fib_E * config.params.fib_A / config.params.fib_L
    m = config.params.fib_m0 + config.params.fib_rho * config.params.fib_A * config.params.fib_L / 3
    F = abs(config.params.fib_BC_Fz) + m * config.params.g
    x_mean = F / k
    w_0 = np.sqrt(k / m)
    print("w_0 =", w_0)
    c_damp = config.params.fib_eta * config.params.fib_A / config.params.fib_L
    zeta = c_damp / (2. * m)
    w_d = np.sqrt(w_0**2 - zeta**2)
    x_mean_l = np.array(len(x) * [x_mean])
    x_plus = np.array([x_mean*(1.+np.exp(-zeta*t)) for t in time])
    x_minus = np.array([x_mean*(1.-np.exp(-zeta*t)) for t in time])

    def local_maxima(x1, x2, x3):
        if x1 < x2 and x3 < x2:
            return True
        else:
            return False

    def local_minima(x1, x2, x3):
        if x1 > x2 and x3 > x2:
            return True
        else:
            return False

    n_max = 0
    n_min = 0
    maxima = []
    minima = []
    for i in range(1, len(x)-1):
        x_previous = x[i-1]
        x_current = x[i]
        x_next = x[i+1]
        if local_maxima(x_previous, x_current, x_next):
            n_max += 1
            maxima.append([x_current, time[i]])
        elif local_minima(x_previous, x_current, x_next):
            n_min += 1
            minima.append([x_current, time[i]])

    error_plus = 0
    error_minus = 0
    for i in range(n_max):
        error_plus += ((maxima[i][0] - x_mean*(1.+np.exp(-zeta*maxima[i][1]))) / x_mean)**2
    for i in range(n_min):
        error_minus += ((minima[i][0] - x_mean*(1.-np.exp(-zeta*minima[i][1]))) / x_mean)**2

    error_rms_x_plus = np.sqrt((1. / n_max) * error_plus)
    error_rms_x_minus = np.sqrt((1. / n_min) * error_minus)

    periods = []
    for i in range(1, n_min):
        t = minima[i][1] - minima[i-1][1]
        periods.append(t)
    period_sim = sum(periods) / len(periods)
    w_d_sim = 2 * np.pi / period_sim
    error_w_d = (w_d_sim - w_d) / w_d

    print("###################### ERRORS ######################\n")
    print("n_max = ", n_max)
    print("n_min = ", n_min)
    print("w_d = ", w_d, "w_d_sim = ", w_d_sim)
    print("n = ", config.params.fib_n,": error_rms_x_plus = ", error_rms_x_plus)
    print("n = ", config.params.fib_n,": error_rms_x_minus = ", error_rms_x_minus)
    print("n = ", config.params.fib_n,": error_w_d = ", error_w_d)
    print("Underdamped harmonic oscillator (c < 2*sqrt(k·m)): ", c_damp, " < ", 2*np.sqrt(k*m))

    fig, ax = plt.subplots()
    #plt.title('Spider position')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')
    plt.plot(time, x, color='b')
    plt.plot(time, x_plus, color='g', linestyle='dashed')
    plt.plot(time, x_minus, color='r', linestyle='dashed')
    plt.plot(time, x_mean_l, color='k', linestyle='dashed')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.rcParams['lines.linewidth'] = 0.1
    string = "./png_files/spider_position_lin_" + str(config.params.fib_n) + ".png"
    fig.savefig(string)


def visualize_lindyn(context):
    x = []
    time = []
    Fz = []
    stringfile = "./csv_files/visualization_dyn_" + str(int(config.params.fib_eta/1000)) + "_" + str(config.params.fib_w_forced_ratio) + ".csv"
    with open(stringfile, newline='') as csvfile:
        visualreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in visualreader:
            x.append(float(row[0]))
            time.append(float(row[1]))
            Fz.append(float(row[2]) * 10**(-7))

    k = config.params.fib_E * config.params.fib_A / config.params.fib_L
    m = config.params.fib_m0 + config.params.fib_rho * config.params.fib_A * config.params.fib_L / 3
    F = abs(config.params.fib_BC_Fz) + m * config.params.g
    w_0 = np.sqrt(k / m)
    print("w_0 =", w_0)
    print("w_forced_ratio =", config.params.fib_w_forced_ratio)
    c_damp = config.params.fib_eta * config.params.fib_A / config.params.fib_L
    zeta = c_damp / (2. * np.sqrt(k * m))
    r = config.params.fib_w_forced / w_0    # resonance at r = 1 approx.
    amp = (abs(config.params.fib_BC_Fz) / k) / ((1 - r**2)**2 + (2 * zeta * r)**2)
    print("amp =", amp)
    if r != 1.0:
        ph_sh = np.arctan(-2 * zeta * r / (1 - r**2))
        print("ph_sh =", ph_sh)

    def local_maxima(x1, x2, x3):
        if x1 < x2 and x3 < x2:
            return True
        else:
            return False

    def local_minima(x1, x2, x3):
        if x1 > x2 and x3 > x2:
            return True
        else:
            return False

    n_max = 0
    n_min = 0
    maxima = []
    minima = []
    print("length x =", len(x))
    for i in range(10, len(x)-1):
        x_previous = x[i-1]
        x_current = x[i]
        x_next = x[i+1]
        if local_maxima(x_previous, x_current, x_next):
            n_max += 1
            maxima.append([x_current, time[i]])
        elif local_minima(x_previous, x_current, x_next):
            n_min += 1
            minima.append([x_current, time[i]])

    """error_plus = 0
    error_minus = 0
    for i in range(n_max):
        error_plus += ((maxima[i][0] - x_mean*(1.+np.exp(-zeta*maxima[i][1]))) / x_mean)**2
    for i in range(n_min):
        error_minus += ((minima[i][0] - x_mean*(1.-np.exp(-zeta*minima[i][1]))) / x_mean)**2

    error_rms_x_plus = np.sqrt((1. / n_max) * error_plus)
    error_rms_x_minus = np.sqrt((1. / n_min) * error_minus)

    periods = []
    for i in range(1, n_min):
        t = minima[i][1] - minima[i-1][1]
        periods.append(t)
    period_sim = sum(periods) / len(periods)
    w_d_sim = 2 * np.pi / period_sim
    error_w_d = (w_d_sim - w_d) / w_d

    print("###################### ERRORS ######################\n")
    print("n_max = ", n_max)
    print("n_min = ", n_min)
    print("w_d = ", w_d, "w_d_sim = ", w_d_sim)
    print("n = ", config.params.fib_n,": error_rms_x_plus = ", error_rms_x_plus)
    print("n = ", config.params.fib_n,": error_rms_x_minus = ", error_rms_x_minus)
    print("n = ", config.params.fib_n,": error_w_d = ", error_w_d)
    print("Underdamped harmonic oscillator (c < 2*sqrt(k·m)): ", c_damp, " < ", 2*np.sqrt(k*m))
    """

    print("n_max = ", n_max)
    print("n_min = ", n_min, "\n")
    print("fib_n = ", config.params.fib_n)
    print("fib_eta = ", config.params.fib_eta)
    print("fib_w_forced_ratio = ", config.params.fib_w_forced_ratio)

    fig, ax = plt.subplots()
    #plt.title('Spider position')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')
    plt.plot(time, x, color='b')
    #plt.plot(time, Fz, color='r')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.rcParams['lines.linewidth'] = 0.1
    string = "./png_files/spider_position_dyn_" + str(int(config.params.fib_eta/1000)) + "_" + str(config.params.fib_w_forced_ratio) + ".png"
    fig.savefig(string)

def visualize_bend(context):
    x = []
    time = []
    stringfile = "./csv_files/visualization_bend_" + str(config.params.fib_n) + ".csv"
    with open(stringfile, newline='') as csvfile:
        visualreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in visualreader:
            x.append(abs(float(row[0])))
            time.append(float(row[1]))

    k = 48 * config.params.fib_E * config.params.fib_I / (config.params.fib_L**3)
    m = (48 / np.pi**4) * config.params.fib_rho * config.params.fib_L * config.params.fib_A
    q = config.params.g * config.params.fib_rho * config.params.fib_L * config.params.fib_A
    F = abs(config.params.fib_BC_Fz)
    c_damp = 48 * config.params.fib_eta * config.params.fib_I / (config.params.fib_L**3)
    x_mean = F / k
    w_0 = np.sqrt(k / m)
    print("w_0 =", w_0)
    zeta = c_damp / (2. * m)
    w_d = np.sqrt(w_0**2 - zeta**2)
    x_mean_l = np.array(len(x) * [x_mean])
    x_plus = np.array([x_mean*(1.+np.exp(-zeta*t)) for t in time])
    x_minus = np.array([x_mean*(1.-np.exp(-zeta*t)) for t in time])

    def local_maxima(x1, x2, x3):
        if x1 < x2 and x3 < x2:
            return True
        else:
            return False

    def local_minima(x1, x2, x3):
        if x1 > x2 and x3 > x2:
            return True
        else:
            return False

    n_max = 0
    n_min = 0
    maxima = []
    minima = []
    # start from i = 10 to eliminate small oscillation in the first time steps
    # that corrupt the errors (checked from the data file)
    for i in range(10, len(x)-1):
        x_previous = x[i-1]
        x_current = x[i]
        x_next = x[i+1]
        if local_maxima(x_previous, x_current, x_next):
            n_max += 1
            maxima.append([x_current, time[i]])
        elif local_minima(x_previous, x_current, x_next):
            n_min += 1
            minima.append([x_current, time[i]])

    error_plus = 0
    error_minus = 0
    for i in range(n_max):
        error_plus += ((maxima[i][0] - x_mean*(1.+np.exp(-zeta*maxima[i][1]))) / x_mean)**2
    for i in range(n_min):
        error_minus += ((minima[i][0] - x_mean*(1.-np.exp(-zeta*minima[i][1]))) / x_mean)**2

    error_rms_x_plus = np.sqrt((1. / n_max) * error_plus)
    error_rms_x_minus = np.sqrt((1. / n_min) * error_minus)

    periods = []
    for i in range(1, n_min):
        t = minima[i][1] - minima[i-1][1]
        periods.append(t)
    period_sim = sum(periods) / len(periods)
    w_d_sim = 2 * np.pi / period_sim
    error_w_d = (w_d_sim - w_d) / w_d

    print("###################### ERRORS ######################\n")
    print("n_max = ", n_max)
    print("n_min = ", n_min)
    print("w_d = ", w_d, "w_d_sim = ", w_d_sim)
    print("n = ", config.params.fib_n,": error_rms_x_plus = ", error_rms_x_plus)
    print("n = ", config.params.fib_n,": error_rms_x_minus = ", error_rms_x_minus)
    print("n = ", config.params.fib_n,": error_w_d = ", error_w_d)
    print("Underdamped harmonic oscillator (c < 2*sqrt(k·m)): ", c_damp, " < ", 2*np.sqrt(k*m))

    fig, ax = plt.subplots()
    #plt.title('Middle point position')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')
    plt.plot(time, x, color='b')
    plt.plot(time, x_plus, color='g', linestyle='dashed')
    plt.plot(time, x_minus, color='r', linestyle='dashed')
    plt.plot(time, x_mean_l, color='k', linestyle='dashed')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.rcParams['lines.linewidth'] = 0.1
    string = "./png_files/middle_point_position_bend_" + str(config.params.fib_n) + ".png"
    fig.savefig(string)


def check_variables(context):
    c = context
    print("\n#------------- VARIABLES -------------#\n")
    print("x_old:\n", c.x_old)
    print("xdot_old:\n", c.xdot_old)
    print("xdotdot_old:\n", c.xdotdot_old)
    print("r:\n", c.r)
    print("r_abs:\n", c.r_abs)
    print("e_r:\n", c.e_r)
    print("cross_e:\n", c.cross_e)
    print("cross_e_abs:\n", c.cross_e_abs)
    print("e_n:\n", c.e_n)
    print("rdot:\n", c.rdot)
    print("ang:\n", c.ang)
    print("angdot:\n", c.angdot)
    print("kappa:\n", c.kappa)
    print("kappadot:\n", c.kappadot)
    print("s:\n", c.s)
    print("eps:\n", c.eps)
    print("epsdot:\n", c.epsdot)
    print("F_k:\n", c.F_k)
    print("F_d:\n", c.F_d)
    print("M_b:\n", c.M_b)
    print("M_c:\n", c.M_c)
    print("F:\n", c.F)
    print("M:\n", c.M)
    print("Q:\n", c.Q)
    print("D_n:\n", c.D_n)
    print("D_l:\n", c.D_l)
    print("D:\n", c.D)
    print("G:\n", c.G)
    print("z_old:\n", c.z_old)
    print("gfun_old:\n", c.gfun_old)

def check_parameters():
    print("\n#--------------------------- PARAMETERS ---------------------------#\n")
    print("fib_n = ", config.params.fib_n)
    print("fib_rho = ", config.params.fib_rho)
    print("fib_d = ", config.params.fib_d)
    print("fib_L = ", config.params.fib_L)
    print("fib_E = ", config.params.fib_E)
    print("fib_eta = ", config.params.fib_eta)
    print("g = ", config.params.g)
    print("fib_m0 = ", config.params.fib_m0)
    print("fib_n_threads = ", config.params.fib_n_threads)
    print("fib_L0 = ", config.params.fib_L0)
    print("fib_n_plus = ", config.params.fib_n_plus)
    print("fib_A = ", config.params.fib_A)
    print("fib_I = ", config.params.fib_I)
    print("fib_m_other = ", config.params.fib_m_other)
    print("fib_ratioLd = ", config.params.fib_ratioLd)
    print("nu = ", config.params.nu)
    print("dt = ", config.params.dt)
    print("T = ", config.params.T)
    print("fib_BC_Fz = ", config.params.fib_BC_Fz)
    print("L = ", config.params.L)
    print("fib_BC_index_F = ", config.params.fib_BC_index_F, "\n")
