from spectralDNS import config
import numpy as np
import tricubic
import csv
import matplotlib.pyplot as plt
from scipy import optimize
from fibredyn.drag_coeffs import func_Cl_KellerRubinow, func_interpolate_l, func_Cl_f
import math
from scipy.interpolate import RegularGridInterpolator
from shenfun import Array, Function, FunctionSpace, TensorProductSpace, VectorSpace
from mpi4py import MPI
from fibredyn import fibutils_2way

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

""" changed by Pascal Müller to suit multiple fibres"""
def writetofile(c, stringfile):

    #manupulation to make the code work at t=0
    delete=False
    if(len(c.nfev))==0:
        delete=True
        c.nfev.append(0)

    if config.params.validate == True: #write energy too
        if config.params.t == 0: #write first rows
            with open(stringfile, mode='w', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow(['t'] + [config.params.t] + ['x1'] + list(c.x_old[..., 0]) + ['E_kin'] + [c.E_kin[-1]] + ['actual energy'] + [c.E_is[-1]])
                visualwriter.writerow(['t'] + [config.params.t] + ['x2'] + list(c.x_old[..., 1]) + ['E_pot'] + [c.E_pot[-1]] + ['predicted energy'] + [c.E_pr[-1]])
                visualwriter.writerow(['t'] + [config.params.t] + ['x3'] + list(c.x_old[..., 2]) + ['W_f'] + [c.W_f[-1]] + ['difference in energy'] + [c.delta[-1]])
                visualwriter.writerow([])

        else: #append already created file
            with open(stringfile, mode='a+', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow(['t'] + [config.params.t] + ['x1'] + list(c.x_old[..., 0]) + ['E_kin'] + [c.E_kin[-1]] + ['actual energy'] + [c.E_is[-1]])
                visualwriter.writerow(['t'] + [config.params.t] + ['x2'] + list(c.x_old[..., 1]) + ['E_pot'] + [c.E_pot[-1]]  + ['predicted energy'] + [c.E_pr[-1]])
                visualwriter.writerow(['t'] + [config.params.t] + ['x3'] + list(c.x_old[..., 2]) + ['W_f'] + [c.W_f[-1]] + ['difference in enegy'] + [c.delta[-1]])
                visualwriter.writerow([])
    else:
        if config.params.t == 0:
            with open(stringfile, mode='w', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow(['t'] + [config.params.t] + ['x1'] + list(c.x_old[..., 0]) + ['function evaluations'] + [c.nfev[-1]])
                visualwriter.writerow(['t'] + [config.params.t] + ['x2'] + list(c.x_old[..., 1]))
                visualwriter.writerow(['t'] + [config.params.t] + ['x3'] + list(c.x_old[..., 2]))
                visualwriter.writerow([])

        else:
            with open(stringfile, mode='a+', newline='') as csvfile:
                visualwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                visualwriter.writerow(['t'] + [config.params.t] + ['x1'] + list(c.x_old[..., 0]) + ['function evaluations'] + [c.nfev[-1]])
                visualwriter.writerow(['t'] + [config.params.t] + ['x2'] + list(c.x_old[..., 1]))
                visualwriter.writerow(['t'] + [config.params.t] + ['x3'] + list(c.x_old[..., 2]))
                visualwriter.writerow([])

    if delete==True:
        c.nfev.pop(-1)


""" Added functions by Pascal Müller """
def get_involved_points (fib_pos, dx):
    """determine neighbouing grid points
        returns position array in grid spacing units
                and 2D array with enclosing grid points
        input:
        fib_pos: array with fibre joint location in absolute unit
        dx:      array with grid spacing for each direction
    """
    pos = np.zeros(3)
    pos[:] = fib_pos[:]/dx[:]
    pos_min=np.floor(pos)
    pos_max=np.ceil(pos)
    for i in range(fib_pos.size):
        if pos_max[i]==config.params.N[i]:
            pos_max[i]=0
    return (pos, np.array([pos_min, pos_max]))

def calc_component_weight(grid_component, pos_component, dx):
    """evaluate 1D weight function
        return float value of 1D weight functions
        input: grid_component, float or int component of grid point
                pos_component, float or int component of fibre joint location
                dx,            float 1D grid spacing
    """
    return 1-abs(pos_component/dx-grid_component)



def calc_point_weight(current_grid_point, pos):
    """ calculate 3D weight function
        return float value of weigth function
        input: current_grid_point, 1D array with grid point coordinates
                pos,                1D array with fibre joint coordinates in absolute units
    """
    w1=calc_component_weight(current_grid_point[0], pos[0], config.params.dx[0])
    w2=calc_component_weight(current_grid_point[1], pos[1], config.params.dx[1])
    w3=calc_component_weight(current_grid_point[2], pos[2], config.params.dx[2])
    return w1*w2*w3


def interpolate_backcoupling_singlecore(fibcontext,fluidcontext):
    """interpolate the drag forces to the fluid grid nodes
        returns shenfun array (physical space) with source term
    """
    cfib=fibcontext
    cflu=fluidcontext
    Force=Array(cflu.VT)
    for i in range (config.params.fib_n_plus):
        pos, grid_points = get_involved_points (cfib.x_new[i,:], config.params.dx)
        for j in range (2):
            for k in range (2):
                for l in range (2): #iterate over all 8 enclosing grid points
                    current_grid_point = np.array([grid_points[j,0], grid_points[k,1], grid_points[l,2]])
                    w=calc_point_weight(current_grid_point,cfib.x_new[i,:])
                    point_force = -w * cfib.D[i,:]
                    point_force = np.divide(point_force,(config.params.rho*config.params.dx[0]*config.params.dx[1]*config.params.dx[2]))
                    Force[:,int(current_grid_point[0]),int(current_grid_point[1]),int(current_grid_point[2])]+=point_force
    return Force


def adapt_locaton(x_l, nprocs,rem,x):
    """Since the Domain is divided to the different threads along the x axis
    this function matches the x coordinate to the array location of
    the corresponing thread
    returns:    array   int     number of the rank computing this location
                loc     int     x coordinate within that subdomain
    inputs:     x_l     int     minimum subdomain length (x direction)
                nprocs  int     number of ranks working
                rem     int     mod(config.params.N,nprocs)
                x       int     x coordinate that needs to be shifted
    """
    disc=rem*(1+x_l)
    if rem==0:
        array=int(x/x_l)
        loc=x%x_l
    elif disc>x:
        array=int(x/(x_l+1))
        loc=x%(x_l+1)
    else:
        array=rem+int((x-disc)/x_l)
        loc=(x-disc)%x_l
    return (array,int(loc))

def interpolate_backcoupling(fibcontext,fluidcontext, nprocs, fib_per_prc=1, number=0):
    """interpolate the drag forces of one fibre to the fluid grid nodes
        returns list of arrays corresponding to a subdomain
    """
    cfib=fibcontext
    cflu=fluidcontext
    x_l=int(config.params.N[0]/nprocs)    #minimum length for each rank
    rem=config.params.N[0]%nprocs          #tells how many ranks need to have one element more
    Force=[]                                #list to return arrays

    #shape of the the solution arrays for the different ranks
    N_minus=[3, x_l]
    if rem!=0:
        N_plus=[3, x_l+1]
    for i in range(1,len(config.params.N)):
        N_minus.append(config.params.N[i])
        if rem!=0:
            N_plus.append(config.params.N[i])

    #initalise list of arrays for each rank to return
    for i in range (nprocs):
        if i>=rem or rem==0:
            Force.append(np.zeros(N_minus))
        else:
            Force.append(np.zeros(N_plus))

    print("new fibre")
    #iterate over fibre joints
    for i in range (config.params.fib_n_plus):
        pos, grid_points = get_involved_points (cfib.x_new[i,:], config.params.dx)
        for element in pos:
            if element < 0 or element > config.params.N[0]:
                #joint is out of domain, hence solver is aborted
                print("velocity", cfib.xdot_old[i,:], "last velocity", cfib.xdot_last[i,:])
                print("i",i, "pos", pos)
                assert False
        for j in range (2):
            for k in range (2):
                for l in range (2): #iterate over all 8 enclosing grid points
                    current_grid_point = np.array([grid_points[j,0], grid_points[k,1], grid_points[l,2]])
                    w=calc_point_weight(current_grid_point,cfib.x_new[i,:])
                    #in case the fibre is further than the last grid point, the force is added on the other
                    #according to periodic BC
                    #for m in range(2):
                    #    if current_grid_point[m]>=config.params.N[m]:
                    #        current_grid_point[m]-=config.params.N[m]
                    point_force = -w * cfib.D[i,:]
                    point_force = np.divide(point_force,(config.params.rho*config.params.dx[0]*config.params.dx[1]*config.params.dx[2]))
                    array, loc = adapt_locaton(x_l, nprocs, rem, current_grid_point[0])
                    #ensure joint lies within domain
                    assert array>=0
                    assert loc>=0
                    assert current_grid_point[1]>=0
                    assert current_grid_point[2]>=0
                    Force[array][:,loc,int(current_grid_point[1]), int(current_grid_point[2])]+=point_force
    return Force


def get_sourceterm_array(Force, Force_np, nprocs, rank, fib_per_prc, fib_num):
    """list containing sourceterm arrays is scattered
        Force       shenfun array       is updated and will contain result (passed by reference)
        Force_np    list of arrays      contains source term, needs to be scattered
        nprocs      int                 amount of ranks
        rank        int                 rank calling the function
        fib_per_prc int                 amount of fibres on this rank
        fib_num     int                 fibre number of first fibre of this rank
    """
    for i in range(config.params.fib_amt):
        sctr=[]   #initialise list to be scattered
        host=fibutils_2way.allocate_fib(i, nprocs)
        for j in range(nprocs):
            if rank==host:
                current_fib = i - fib_num
                sctr += Force_np[current_fib*nprocs:(current_fib+1)*nprocs]
                break
        sctr=MPI.COMM_WORLD.scatter(sctr, root=host)
        Force[:] += sctr #add up contributions of all fibres


def test_fft(fluidcontext):
    """creates a shenfun array with known fft"""
    u=Array(fluidcontext.VT)
    for i in range (config.params.N[0]):
        for j in range (config.params.N[1]):
            u[0,i,j,:]=np.sin(fluidcontext.V[0].mesh()[i])*np.sin(2* fluidcontext.V[1].mesh()[j])* np.sin(3*fluidcontext.V[2].mesh())
            u[1,i,j,:]=np.cos(fluidcontext.V[0].mesh()[i])*np.cos(2* fluidcontext.V[1].mesh()[j])* np.cos(3*fluidcontext.V[2].mesh())
            u[2,i,j,:]=np.cos(fluidcontext.V[0].mesh()[i])*np.cos(2* fluidcontext.V[1].mesh()[j])* np.cos(3*fluidcontext.V[2].mesh())+np.sin(fluidcontext.V[0].mesh()[i])*np.sin(2* fluidcontext.V[1].mesh()[j])* np.sin(3*fluidcontext.V[2].mesh())
    return u

def add_backcoupling(fluidcontext, Force):
    """Force is fourier transromed and saved in the sourceterm array"""
    fluidcontext.Source=Force.forward()

def writeforce(Force, stringfile1, stringfile2, stringfile3):
    """create a csv file containing sourceterm in physical space"""
    if config.params.tstep == 1:
        m='w'
    else:
        m='a+'
    for i in range(config.params.N[2]):
        with open(stringfile1, mode=m, newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
	                                	quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow(["step=", config.params.tstep, "z=", i])
            visualwriter.writerows(Force[0,:,:,i])
        m='a+'

    if config.params.tstep == 1:
        m='w'
    else:
        m='a+'
    for i in range(config.params.N[2]):
        with open(stringfile2, mode=m, newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
	                                	quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow(["step=", config.params.tstep, "z=", i])
            visualwriter.writerows(Force[1,:,:,i])
        m='a+'

    if config.params.tstep == 1:
        m='w'
    else:
        m='a+'
    for i in range(config.params.N[2]):
        with open(stringfile3, mode=m, newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
	                                	quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow(["step=", config.params.tstep, "z=", i])
            visualwriter.writerows(Force[2,:,:,i])
        m='a+'

def writesource(source, string1='Results/source_u_hat.csv',
                string2='Results/source_v_hat.csv',
                string3='Results/source_w_hat.csv'):
    """create csv file cotaining sourceterm in fourier space"""
    if config.params.tstep == 1:
        m='w'
    else:
        m='a+'
    for i in range(int(config.params.N[2]/2+1)):
        with open(string1, mode=m, newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
	                                	quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow(["step=", config.params.tstep, "z=", i])
            visualwriter.writerows(source[0,:,:,i])
        m='a+'

    if config.params.tstep == 1:
        m='w'
    else:
        m='a+'
    for i in range(int(config.params.N[2]/2+1)):
        with open(string2, mode=m, newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
	                                	quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow(["step=", config.params.tstep, "z=", i])
            visualwriter.writerows(source[1,:,:,i])
        m='a+'

    if config.params.tstep == 1:
        m='w'
    else:
        m='a+'
    for i in range(int(config.params.N[2]/2+1)):
        with open(string3, mode=m, newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
	                                	quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow(["step=", config.params.tstep, "z=", i])
            visualwriter.writerows(source[2,:,:,i])
        m='a+'

def writemomentum(u1,u2,u3,f1,f2,f3):
    """create a csv file with momentum conservation terms"""
    t_list=[]
    x_delta=[]
    y_delta=[]
    z_delta=[]
    for i in range(int(config.params.T/config.params.dt)+1):
        t_list.append(i)
        if i==0:
            x_delta.append(0)
            y_delta.append(0)
            z_delta.append(0)
        else:
            x_delta.append(u1[i]-(u1[i-1]+f1[i-1]))
            y_delta.append(u2[i]-(u2[i-1]+f2[i-1]))
            z_delta.append(u3[i]-(u3[i-1]+f3[i-1]))
    m='w'
    with open('Results/momentum.csv', mode=m, newline='') as csvfile:
            visualwriter = csv.writer(csvfile, delimiter=',',
	                                	quotechar='|', quoting=csv.QUOTE_MINIMAL)
            visualwriter.writerow(["time step"] + t_list)
            visualwriter.writerow(["u1"] + u1)
            visualwriter.writerow(["f1"] + f1)
            visualwriter.writerow(["error in first dimension"] + x_delta)
            visualwriter.writerow(["u2"] + u2)
            visualwriter.writerow(["f2"] + f2)
            visualwriter.writerow(["error in second dimension"] + y_delta)
            visualwriter.writerow(["u3"] + u3)
            visualwriter.writerow(["f3"] + f3)
            visualwriter.writerow(["error in third dimension"] + z_delta)

def plot_momentum(u1,u2,u3,f1,f2,f3,dt):
        """plot conservation of momentum
        """
        assert len(f1)==len(f2)
        assert len(f1)==len(f3)
        assert len(f1)==len(u1)
        assert len(f1)==len(u2)
        assert len(f1)==len(u3)

        X=np.arange(len(f1))*dt
        Y_is_1=np.empty(len(f1))
        Y_pr_1=np.empty(len(f1))
        Y_err_1=np.empty(len(f1))
        Y_is_2=np.empty(len(f1))
        Y_pr_2=np.empty(len(f1))
        Y_err_2=np.empty(len(f1))
        Y_is_3=np.empty(len(f1))
        Y_pr_3=np.empty(len(f1))
        Y_err_3=np.empty(len(f1))

        for i in range(len(f1)):
            Y_is_1[i]=u1[i]
            Y_is_2[i]=u2[i]
            Y_is_3[i]=u3[i]
            if i==0:
                Y_pr_1[0]=u1[0]
                Y_pr_2[0]=u2[0]
                Y_pr_3[0]=u3[0]
            else:
                Y_pr_1[i]=Y_pr_1[i-1]+f1[i-1]
                Y_pr_2[i]=Y_pr_2[i-1]+f2[i-1]
                Y_pr_3[i]=Y_pr_3[i-1]+f3[i-1]

            Y_err_1[i]=abs(Y_is_1[i]-Y_pr_1[i])
            Y_err_2[i]=abs(Y_is_2[i]-Y_pr_2[i])
            Y_err_3[i]=abs(Y_is_3[i]-Y_pr_3[i])

        #plots
        figure,axis=plt.subplots(1,3, sharex = True, sharey = False, figsize=(12,4))
        axis[0].plot(X,Y_is_1, label='actual velocity sum')
        axis[0].plot(X,Y_pr_1, label='predicted velocity sum', linestyle=':', linewidth=3)
        axis[0].plot(X,Y_err_1, label='difference')

        axis[1].plot(X,Y_is_2, label='actual velocity sum')
        axis[1].plot(X,Y_pr_2, label='predicted velocity sum', linestyle=':', linewidth=3)
        axis[1].plot(X,Y_err_2, label='difference')

        axis[2].plot(X,Y_is_3, label='actual velocity sum')
        axis[2].plot(X,Y_pr_3, label='predicted velocity sum', linestyle=':', linewidth=3)
        axis[2].plot(X,Y_err_3, label='difference')

        axis[0].set_title("First Axis (x)")
        axis[1].set_title("Second Axis (y)")
        axis[2].set_title("Third Axis(z)")

        for i in range(3):
            axis[i].set_xlabel('time')
            axis[i].set_ylabel('sum of all velocities')
            #axis[i].legend()
        axis[1].legend()
        fname = "Results/momentum_N_"+str(config.params.N[0])+"_"+str(config.params.N[1])+"_"+str(config.params.N[2])+"fib_amt"+str(config.params.fib_amt)+"_dt_"+str(config.params.dt)+"_T_"+str(config.params.T)+".png"
        plt.savefig(fname=fname)

def plot_energy(context,dt):
    """plot energy conervation terms
        only fibers from rank 0 will be plotted
    """
    X=np.arange(len(context[0].E_is))*dt
    Y_is=[]
    Y_pr=[]
    Y_diff=[]
    Y_w=[]
    for i in range(len(context)):
        Y_is.append(np.empty(len(context[0].E_is)))
        Y_pr.append(np.empty(len(context[0].E_is)))
        Y_diff.append(np.empty(len(context[0].E_is)))
        Y_w.append(np.empty(len(context[0].E_is)))
        for j in range(len(Y_is[i])):
            if j==0:
                Y_is[i][j]=0
            else:
                Y_is[i][j]=context[i].E_is[j]-context[i].E_is[j-1]
            Y_pr[i][j]=context[i].W_f[j]
            Y_diff[i][j]=abs(context[i].E_is[j]-context[i].E_pr[j])
            Y_w[i][j]=abs(context[i].W_f[j])
    size=len(context)*5
    figure, axis=plt.subplots(len(context),2, sharex=True, sharey=False, figsize=(10,size), squeeze=False)
    for i in range(len(context)):
        axis[i,0].plot(X,Y_is[i], label='actual energy change')
        axis[i,0].plot(X,Y_pr[i], label='predicted energy change', linestyle=':', linewidth=3)
        axis[i,0].set_title("Fibre No "+str(i)+" Predicted and Actual Energy", pad=15)
        axis[i,0].set_xlabel("time")
        axis[i,0].set_ylabel("added energy")
        axis[i,0].legend()

        axis[i,1].semilogy(X[1:],Y_diff[i][1:], label='prediction error')
        axis[i,1].semilogy(X[1:],Y_w[i][1:], label='Stokesian work')
        axis[i,1].set_title("Fibre No "+str(i)+" Error and Work")
        axis[i,1].set_xlabel("time")
        axis[i,1].set_ylabel("work")
        axis[i,1].legend()
    plt.subplots_adjust(hspace=0.5)
    fname = "Results/energy_N_"+str(config.params.N[0])+"_"+str(config.params.N[1])+"_"+str(config.params.N[2])+"fib_amt"+str(config.params.fib_amt)+"_dt_"+str(config.params.dt)+"_T_"+str(config.params.T)+".png"
    plt.savefig(fname=fname)

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
