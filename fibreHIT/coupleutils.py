from spectralDNS import config
import numpy as np
import tricubic
import csv
import matplotlib.pyplot as plt
from scipy import optimize
from fibredyn.drag_coeffs import func_Cl_KellerRubinow, func_interpolate_l, func_Cl_f
import math
from scipy.interpolate import RegularGridInterpolator

def interpolate_fibre_v2(fibcontext, fluidcontext, x, y, z, V1, V2, V3):
    fn1 = RegularGridInterpolator((x,y,z), V1)
    fn2 = RegularGridInterpolator((x,y,z), V2)
    fn3 = RegularGridInterpolator((x,y,z), V3)

    for i, x in enumerate(fibcontext.x_old):
        for comp in range(3):
            if x[comp] > config.params.L[comp] or x[comp] < 0:
                x[comp] = x[comp] % config.params.L[comp]
        fibcontext.u_ip[i,0] = fn1(x) #interpolate the function f at a point in space
        fibcontext.u_ip[i,1] = fn2(x)
        fibcontext.u_ip[i,2] = fn3(x)
        # Calculate relative fluid velocities at each mass point location:
        fibcontext.v_rel[i, ...] = fibcontext.u_ip[i, ...] - fibcontext.xdot_old[i, ...]
        fibcontext.Re.append(np.linalg.norm(fibcontext.v_rel[i, ...]) * config.params.fib_d / config.params.nu)


def interpolate_fibre(fibcontext, fluidcontext):
    """Interpolate the fluid velocities to the mass point positions"""

    # fluidcontext.U
    n_ip_1 = config.params.N[0] + 1
    n_ip_2 = config.params.N[1] + 1
    n_ip_3 = config.params.N[2] + 1
    N_ip = [n_ip_1, n_ip_2, n_ip_3]
    #some function f(x,y,z) is given on a cubic grid indexed by i,j,k:
    f1 = np.zeros((n_ip_1, n_ip_2, n_ip_3), dtype='float')   # first fluid velocity components
    f2 = np.zeros((n_ip_1, n_ip_2, n_ip_3), dtype='float')   # second fluid velocity components
    f3 = np.zeros((n_ip_1, n_ip_2, n_ip_3), dtype='float')   # third fluid velocity components
    for i in range(n_ip_1):
        inew = 0 if i == n_ip_1-1 else i
        for j in range(n_ip_2):
            jnew = 0 if j == n_ip_2-1 else j
            for k in range(n_ip_3):
                knew = 0 if k == n_ip_3-1 else k
                f1[i][j][k] = fibcontext.u1[inew, jnew, knew]
                f2[i][j][k] = fibcontext.u2[inew, jnew, knew]
                f3[i][j][k] = fibcontext.u3[inew, jnew, knew]

    #initialize interpolator with input data on cubic grid
    ip1 = tricubic.tricubic(list(f1), N_ip)
    ip2 = tricubic.tricubic(list(f2), N_ip)
    ip3 = tricubic.tricubic(list(f3), N_ip)

    # LOOP OVER REQUIRED VELOCITIES - loop over positions of mass points i = 1, ..., n_plus-1
    for i, x in enumerate(fibcontext.x_old):
        for comp in range(3):
            if x[comp] > config.params.L[comp] or x[comp] < 0:
                x[comp] = x[comp] % config.params.L[comp]
        fibcontext.u_ip[i,0] = ip1.ip(list(x*(config.params.N[0]/config.params.L[0]))) #interpolate the function f at a point in space
        fibcontext.u_ip[i,1] = ip2.ip(list(x*(config.params.N[1]/config.params.L[1])))
        fibcontext.u_ip[i,2] = ip3.ip(list(x*(config.params.N[2]/config.params.L[2])))
        # Calculate relative fluid velocities at each mass point location:
        fibcontext.v_rel[i, ...] = fibcontext.u_ip[i, ...] - fibcontext.xdot_old[i, ...]
        fibcontext.Re.append(np.linalg.norm(fibcontext.v_rel[i, ...]) * config.params.fib_d / config.params.nu)
    #print(fibcontext.u_ip)
    #print(fibcontext.v_rel)
    #print(fibcontext.xdot_old)

def compute_terminal_velocity():
    g = abs(config.params.g)
    rho_f = config.params.rho
    rho_p = config.params.fib_rho   # same density for spider and silk
    d = config.params.fib_d     # diameter fiber (silk)
    nu_f = config.params.nu
    L = config.params.fib_L
    mp = config.params.fib_m0
    dp = (24*mp/(4*math.pi*rho_p))**(1/3)   # diameter particle (spider)
    tau_St = dp**2 * rho_p / (18 * rho_f * nu_f)
    def fun(vp):
        Re = d*vp/nu_f
        if Re < 0.1:
            C_l = func_Cl_KellerRubinow(Re, 0, L/d)
        elif Re >= 0.1 and Re < 10.:
            C_l = func_interpolate_l(Re, 0, L/d)
        elif Re >= 10.:
            C_l = func_Cl_f(Re, 0)
        f_vp = g*(rho_p-rho_f)/rho_p - vp*(1+0.15*(dp*vp/nu_f)**0.687)/tau_St - 2*rho_f*C_l*vp**2 / (math.pi*d*rho_p)
        return f_vp
    def funprime(vp):
        Re = d*vp/nu_f
        if Re < 0.1:
            C_l = func_Cl_KellerRubinow(Re, 0, L/d)
        elif Re >= 0.1 and Re < 10.:
            C_l = func_interpolate_l(Re, 0, L/d)
        elif Re >= 10.:
            C_l = func_Cl_f(Re, 0)
        fprime_vp = -1/tau_St - 0.15*vp**0.687 *(dp/nu_f)**0.687 /tau_St - 4*rho_f*C_l*vp / (math.pi*d*rho_p)
        return fprime_vp
    result = optimize.root_scalar(fun, x0=0.1, fprime=funprime, method="newton", xtol=1e-12)
    return result.root, result.iterations, result.function_calls

def out_of_domain():
    pass

def plot_isotropic_slice(c, sol, str_animation):
    #X = c.X.get((slice(None), slice(None), slice(None), slice(None)))
    X = [np.zeros(tuple(config.params.N)), np.zeros(tuple(config.params.N)), np.zeros(tuple(config.params.N))]
    Xglobal = c.T.mesh()
    for i in range(config.params.N[0]):
        for j in range(config.params.N[1]):
            for k in range(config.params.N[2]):
                X[0][i, j, k] = Xglobal[0][i, 0, 0]
                X[1][i, j, k] = Xglobal[1][0, j, 0]
                X[2][i, j, k] = Xglobal[2][0, 0, k]

    U = c.U.get((slice(None), slice(None), slice(None), slice(None)))
    if sol.rank == 0:
        plt.figure(config.params.tstep)
        #im1 = plt.contourf(c.X[1][:,:,0], c.X[0][:,:,0], div_u[:,:,10], 100)
        plt.contourf(X[1][..., 0], X[0][..., 0], U[0, ..., 10], 100)
    # Added by Joan:
        plt.savefig(str_animation)
        plt.close()

def writetofile(c, stringfile1, stringfile2, stringfile3):

    if config.params.tstep == 1:
        with open(stringfile1, mode='w', newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow([config.params.t] + list(c.x_new[..., 0]) + ["Drag_Force"] + list(c._D[...,0]))

        with open(stringfile2, mode='w', newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow([config.params.t] + list(c.x_new[..., 1]) + ["Drag_Force"] + list(c._D[...,1]))

        with open(stringfile3, mode='w', newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow([config.params.t] + list(c.x_new[..., 2]) + ["Drag_Force"] + list(c._D[...,2]))

    else:
        with open(stringfile1, mode='a+', newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow([config.params.t] + list(c.x_new[..., 0]) + ["Drag_Force"] + list(c._D[...,0]))

        with open(stringfile2, mode='a+', newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow([config.params.t] + list(c.x_new[..., 1]) + ["Drag_Force"] + list(c._D[...,1]))

        with open(stringfile3, mode='a+', newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow([config.params.t] + list(c.x_new[..., 2]) + ["Drag_Force"] + list(c._D[...,2]))

def check_parameters(sol):
    print("\n#--------------------------- PARAMETERS ---------------------------#\n")
    print("General solver parameters:\n")
    print(" coupled_solver = ", config.params.coupled_solver)
    print(" init_from_file = ", config.params.init_from_file)
    print(" checkpoint = ", config.params.checkpoint)
    print(" write_result = ", config.params.write_result)
    print(" compute_energy = ", config.params.compute_energy)
    print(" compute_spectrum = ", config.params.compute_spectrum)
    print(" plot_step = ", config.params.plot_step)
    print(" dealias = ", config.params.dealias)
    print(" fib_fixed = ", config.params.fib_fixed)
    print(" dt = ", config.params.dt)
    print(" T = ", config.params.T)
    print(" L = ", config.params.L, "\n")

    print("Fibre dynamics parameters:\n")
    print(" fib_n = ", config.params.fib_n)
    print(" fib_rho = ", config.params.fib_rho)
    print(" fib_d = ", config.params.fib_d)
    print(" fib_L = ", config.params.fib_L)
    print(" fib_E = ", config.params.fib_E)
    print(" fib_eta = ", config.params.fib_eta)
    print(" g = ", config.params.g)
    print(" fib_m0 = ", config.params.fib_m0)
    print(" fib_n_threads = ", config.params.fib_n_threads)
    print(" fib_L0 = ", config.params.fib_L0)
    print(" fib_n_plus = ", config.params.fib_n_plus)
    print(" fib_A = ", config.params.fib_A)
    print(" fib_I = ", config.params.fib_I)
    print(" fib_m_other = ", config.params.fib_m_other)
    print(" fib_start = ", config.params.fib_start)
    print(" fib_axis = ", config.params.fib_axis)
    print(" fib_ratioLd = ", config.params.fib_ratioLd, "\n")

    print("Homogeneous isotropic turbulence parameters:\n")
    print(" nu = ", config.params.nu)
    print(" N = ", config.params.N)
    print(" Kf2 = ", config.params.Kf2)
    print(" kd = ", config.params.kd)
    print(" Re_lam = ", config.params.Re_lam)
    print(" rho = ", config.params.rho, "\n")

    print("MPI:\n")
    print(" number of processes = ", sol.num_processes, "\n")

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
    print("v_rel:\n", c.v_rel)
    #print("u1:\n", c.u1)
    #print("u2:\n", c.u2)
    #print("u3:\n", c.u3)
    print("u_ip:\n", c.u_ip)
