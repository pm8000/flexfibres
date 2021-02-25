"""
2 way coupled fibre dynamics - spectralDNS solver
Joan Clapes Roig and Pascal Müller
"""

from __future__ import print_function
import warnings
import math
import time
import statistics
from spectralDNS import config, get_solver
from spectralDNS.utilities import Timer
from Isotropic import initialize, L2_norm, spectrum, update, init_from_file
from FibreSolver import get_fibcontext, initialize_fibre, initialize_fibre_multi
from fibredyn import fibutils_2way
from fibreHIT import coupleutils_2way
import numpy as np
import h5py     # Reference: http://docs.h5py.org/en/stable/
#import pdb     # -> Python Debugger
import check_drag
from mpi4py import MPI
from shenfun import Array

if __name__ == "__main__":

    ###----------------------------- PARAMETERS -----------------------------###


    # Configure coupled solver parameters defined in spectralDNS/config.py:
    config.update(
        {'coupled_solver': True,
         'fib_n': 20,
         'fib_rho': 1320.,
         'fib_d': 0.231e-3,
         'fib_L': 3.2,
         'fib_E': 13.8e9,
         'fib_eta': 5.0e3,
         'g': -9.80665,
         'fib_m0': 0.018,    # modified from 0.018
         'fib_n_threads': 1,
         'fib_L0': 1.,
         'fib_n_plus': 1,
         'fib_A': 1.,
         'fib_I': 1.,
         'fib_m_other': 1.,
         'fib_ratioLd': 1.,
         'fib_fixed': False,
         'nu': 0.005428,
         'dt': 0.001,
         'T': 2.,
         'L': [2.*np.pi, 2.*np.pi, 2.*np.pi],
         'checkpoint': 1000000,
         'write_result': 1000000,
         'dealias': '3/2-rule',
        }, "triplyperiodic"
    )

    # Homogeneous Isotropic Turbulence parameters:
    #
    #       k0 = 2·pi / L = 1   (because domain L = 2·pi)
    #       kf/k0 = 3           (because domain k0 = 1) -> kf same as kf2
    #       kd/k0 = 50          (because domain k0 = 1)
    #       nu =  5.428e-03     (because domain k0 = 1)
    #       Re = 43             (because domain h = 1)
    #       Re_lam = 84         (because domain h = 1)
    #       h = hyperviscosity index
    #       k = turbulent kinetic energy

    # Additional arguments of the spectralDNS HIT Solver:
    config.triplyperiodic.add_argument("--N",
        default=[128, 128, 128], nargs=3, help="Mesh size. Trumps M.")
    config.triplyperiodic.add_argument("--compute_energy", type=int,
        default=1000)
    config.triplyperiodic.add_argument("--compute_spectrum", type=int,
        default=1000)
    config.triplyperiodic.add_argument("--plot_step", type=int,
        default=1000000)
    config.triplyperiodic.add_argument("--Kf2", type=int,
        default=3)   # (specified) largest wavenumber acted upon by the forcing
    config.triplyperiodic.add_argument("--kd", type=float,
        default=50.)    #  hyperviscous dissipation wavenumber
    config.triplyperiodic.add_argument("--Re_lam", type=float,
        default=84.)    # microscale Reynolds number, Re_lambda
    config.triplyperiodic.add_argument("--rho", type=float,
        default=1.299, help='Density of the fluid, [rho]=kg/m^3')
    config.triplyperiodic.add_argument("--init_from_file", type=bool,
        default=False, help='Initialize flow field from a previous simulation')
                # Density of fluid not necessary

    # Implement additional fiber dynamics solver arguments:
    config.triplyperiodic.add_argument("--fib_start",
        default=[2*np.pi/4, np.pi, np.pi], nargs=3,
        help="Initial position of the 0^th mass point.")
    config.triplyperiodic.add_argument("--fib_axis", type=float,
        default=1, choices=(1, 2, 3),
        help="Axis along which the fiber extends: 1 (x), 2 (y), 3 (z)")
    config.triplyperiodic.add_argument("--fib_BC_Fz", type=float,
        default=-10000., help="[N]")
    config.triplyperiodic.add_argument("--fib_BC_index_F", type=int,
        default=0, help="Index of the mass point to which the force is applied.")
    config.triplyperiodic.add_argument("--fib_test", type=str,
        default='notest', choices=('linstat', 'lindyn', 'bend', 'notest'),
        help="Which test case for validation.")
    config.triplyperiodic.add_argument("--fib_w_forced", type=float,
        default=1., help="Forcing frequency of the harmonic oscillator [rad/s]")
    config.triplyperiodic.add_argument("--termvel", type=bool,
        default=False, help="Simulate spider in terminal velocity conditions")
    config.triplyperiodic.add_argument("--fib_amt", type=int, default=1,
        help="amount of fibres to be randomly immersed in the fluid")
    config.triplyperiodic.add_argument("--no_spider", type=bool, default=False,
        help="have fibres with free ends")
    config.triplyperiodic.add_argument("--validate", type=bool, default=False,
        help="fluid conservation of momentum and fibre conservation of energy is computed, fibre segments are not joint, turbulence is not retained, init_from_file must be True")
    config.triplyperiodic.add_argument("--write_fib", type=int, default=1,
        help="each nth time step the fibre position will be saved")

    ###---------------- FLUID SOLVER, ADDITIONAL PARAMETERS ----------------###

    # Passing the function 'update()' defined above (in Isotropic.py):
    fluid_solver = get_solver(update=update, mesh="triplyperiodic")  # see __init__.py

    # Additional flow parameters:
    if config.params.nu==0.005428:
        config.params.nu = (1./config.params.kd**(4./3.))

    # Additional Fibre Dynamics parameters:
    config.params.fib_L0 = config.params.fib_L / config.params.fib_n
    config.params.fib_n_plus = config.params.fib_n + 1
            # n_plus: number of mass points including th spider
    config.params.fib_n_threads = 1
    config.params.fib_A = math.pi * config.params.fib_d**2 / 4
    config.params.fib_I = math.pi * config.params.fib_d**4 / 64
    config.params.fib_ratioLd = config.params.fib_L / config.params.fib_d
    config.params.fib_m_other = config.params.fib_rho * config.params.fib_L *\
                                    config.params.fib_A / config.params.fib_n
    if config.params.no_spider==True:
        config.params.fib_m0=config.params.fib_m_other
    #----------------------------------
    # Parameter coupled_solver indicates the activation of the drag forces:
    config.params.coupled_solver = True
    #----------------------------------

    if fluid_solver.rank == 0:
        print("\n################## FIBRE DYNAMICS - spectralDNS SOLVER ##################\n")

    ###--------------------- CONTEXT AND INITIALIZATION ---------------------###

    # Get the fluid and fiber contexts to solve the fibre dynamics:
    fluid_context = fluid_solver.get_context()
    #distribute the fibres to the processes
    fib_per_prc=fibutils_2way.distribute_fibres(fluid_solver.num_processes, fluid_solver.rank)
    fib_num=fibutils_2way.fib_num(fluid_solver.rank, fluid_solver.num_processes, fib_per_prc)
    fiber_context = []
    for i in range(fib_per_prc):
        #get fiber context for each fibre
        fiber_context.append(get_fibcontext())
    # initialize fluid flow and fiber dynamics:
    initialize(fluid_solver, fluid_context)

    # NOTE: The contexts are dictionaries of the class AttributeDict

    # Check the values of all the parameters defined for the fiber dynamics:
    if fluid_solver.rank == 0:
        coupleutils_2way.check_parameters(fluid_solver)

    # HDF5 file name:
    #filestr = "Results/NS_coupled_isoDNS_{}_{}_{}_T{}"
    #fluid_context.hdf5file.filename = filestr.format(*config.params.N, config.params.T)
    #filestr = "Results/NS_coupled_isoDNS_{}_{}_{}_dt{}"
    #fluid_context.hdf5file.filename = filestr.format(*config.params.N, config.params.dt)
    filestr = "Results/NS_coupled_isoDNS_{}_{}_{}_fibn{}_dt{}_T{}"
    fluid_context.hdf5file.filename = filestr.format(*config.params.N, config.params.fib_n, config.params.dt, config.params.T)

    # Fibre I/O:
    stringfile = []
    for i in range(fib_per_prc):
        if config.params.termvel:
            stringfile.append("./Results/fiber_#" + str(fib_num+i) + "_" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + "_TV.csv")
        else:
            stringfile.append("./Results/fiber_#" + str(fib_num+i) + "_" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + ".csv")
    # Isotropic plot:
    str_animation_final = "NS_visualize/Isotropic_" + str(config.params.tstep) + "_fn" + str(config.params.fib_n) + "_N" + str(config.params.N) + "_dt" + str(config.params.dt) + ".png"
    str_animation_init = "NS_visualize/Isotropic_init_" + str(config.params.tstep) + "_fn" + str(config.params.fib_n) + "_N" + str(config.params.N) + "_dt" + str(config.params.dt) + ".png"

    ### Not called at the beggining! Called afterwards when willing to use the
    ### file as initial condition (previous simulation required):
    if config.params.init_from_file == True:
        fileinitstr = "NS_isotropic_{}_{}_{}".format(*config.params.N) + "_c.h5"
        init_from_file(fileinitstr, fluid_solver, fluid_context)

    #get initial velocity, will be reduced to rank 0
    fluid_solver.get_velocity(**fluid_context)
    u1_init = fluid_context.U.get((0, slice(None), slice(None), slice(None)))
    u2_init = fluid_context.U.get((1, slice(None), slice(None), slice(None)))
    u3_init = fluid_context.U.get((2, slice(None), slice(None), slice(None)))

    #unitialise array on other ranks
    if fluid_solver.rank!=0:
        u1_init = np.zeros(fluid_context.U.global_shape)
        u2_init = np.zeros(fluid_context.U.global_shape)
        u3_init = np.zeros(fluid_context.U.global_shape)

    #calculate initial momentum
    if config.params.validate == True and fluid_solver.rank==0:
        u1_mom_new=0
        u2_mom_new=0
        u3_mom_new=0
        for i in range(config.params.N[0]):
            for j in range(config.params.N[1]):
                for k in range(config.params.N[2]):
                    u1_mom_new += u1_init[i,j,k]
                    u2_mom_new += u2_init[i,j,k]
                    u3_mom_new += u3_init[i,j,k]
        u1_mom=[u1_mom_new]
        u2_mom=[u2_mom_new]
        u3_mom=[u3_mom_new]
        f1=[0]
        f2=[0]
        f3=[0]

    #broadcast initial velocity to all ranks
    u1_init=MPI.COMM_WORLD.bcast(u1_init,root=0)
    u2_init=MPI.COMM_WORLD.bcast(u2_init,root=0)
    u3_init=MPI.COMM_WORLD.bcast(u3_init,root=0)
    #initialise fibres with initial velocity
    for i in range(fib_per_prc):
        if config.params.termvel == False:
            initialize_fibre_multi(fiber_context[i], u1=u1_init, u2=u2_init, u3=u3_init)
        else:
            terminal_velocity, iterations, function_calls = coupleutils.compute_terminal_velocity()
            if fluid_solver.rank == 0:
                print("\nTerminal velocity by the Newton method: {} m/s\nIterations: {}\nFunction_calls: {}\n".format(terminal_velocity, iterations, function_calls))
                terminal_velocity = - terminal_velocity
            initialize_fibre_multi(fiber_context[i], u1=u1_init, u2=u2_init, u3=u3_init)
        if config.params.validate == True:
            fibutils_2way.compute_fib_energy(fiber_context[i])
    """
    # Data for the 'NS_coupled_isoDNS_{}_{}_{}_T{}.h5' file:
    Ek, bins, E0, E1, E2 = spectrum(fluid_solver, fluid_context)
    fluid_context.spectrumname = fluid_context.hdf5file.filename+".h5"
    f = h5py.File(fluid_context.spectrumname, mode='w', driver='mpio', comm=fluid_solver.comm)
    f.create_group("Turbulence")
    f["Turbulence"].create_group("Ek")
    bins = np.array(bins)
    f["Turbulence"].create_dataset("bins", data=bins)
    f.close()
    """
    for i in range(fib_per_prc):
        coupleutils_2way.writetofile(fiber_context[i], stringfile[i])

    ###------------------------- ITERATIVE PROCESS --------------------------###
    if fluid_solver.rank == 0:
        print("\n#----------------------- ITERATIVE PROCESS ----------------------#\n")

    #fibutils.check_variables(fib_context)

    #coupleutils.plot_isotropic_slice(fluid_context, fluid_solver, str_animation_init)

    # Iterative process in time to solve the homogeneous isotropic turbulent flow:
    fluid_solver.timer = fluid_solver.Timer()

    # Overloads the function conv() defined in spectralinit.py for NS.py
    fluid_solver.conv = fluid_solver.getConvection(config.params.convection)

    # Gets integrator function. Defined in spectralinit.py based on NS.py
    integrate = fluid_solver.getintegrator(fluid_context.dU, # rhs array
                                     fluid_context.u,  # primary variable
                                     fluid_solver,
                                     fluid_context)

    # Set further parameters to call the functions:
    n = config.params.fib_n
    n_plus = config.params.fib_n_plus
    test = config.params.fib_test
    Fz_dyn = config.params.fib_BC_Fz

    # fluid_solver.comm <- MPI.COMM_WORLD
    nprocs = fluid_solver.num_processes # <- comm.Get_size()
    rank = fluid_solver.rank # <- comm.Get_rank()

    dt_in = config.params.dt

    # Time loop:
    # Initially: config.params.t = 0.0 -> see spectralDNS.config
    #x vectors contain one more element to enable interpolation of last cell (u(x_0)=u(x_N))

    x1 = np.linspace(0, config.params.L[0], config.params.N[0] + 1)
    x2 = np.linspace(0, config.params.L[1], config.params.N[1] + 1)
    x3 = np.linspace(0, config.params.L[2], config.params.N[2] + 1)
    if fluid_solver.rank==0:
        print("Grid spacing: ", config.params.dx)

    while config.params.t + config.params.dt <= config.params.T+1e-12:
        t_fluid_0 = time.time()
        u, config.params.dt, dt_took = integrate()
        #u_fw=Array(fluid_context.VT)
        #u_fw=fluid_context.VT.backward(fluid_context.u, u_fw)

        config.params.t += dt_took
        config.params.tstep += 1

        # Print to track the evolution of the simulation:
        if fluid_solver.rank == 0:
            print("\nTime step: ", config.params.tstep)

        #---------------------# Solve for the fluid flow #---------------------#
        # Solve fluid flow:
        fluid_solver.update(fluid_context, backcoupling=True, validate=config.params.validate)

        if rank == 0:
            print('1st: Fluid computed in {:.3f} sec'.format(time.time() - t_fluid_0))

        # The function update() calculates the velocity field in cartesian
        # coordinates at every position in the mesh only if needed, but in order
        # to calculate the drag at every time step we need to compute the velocity
        # field at every timestep:
        t_getvelocity_0 = time.time()
        fluid_solver.get_velocity(**fluid_context)
        if rank == 0:
            print('2nd: Get velocity field computed in {:.3f} sec'.format(time.time() - t_getvelocity_0))

        # Parallel file I/O (..._w.h5 and ..._c.h5), fluid variables:
        fluid_context.hdf5file.update(config.params, **fluid_context)

        # get() is a collective operation, need to be called by all ranks. The
        # arrays are reduced to rank 0
        t_communication_0 = time.time()
        u1_no_BC = fluid_context.U.get((0, slice(None), slice(None), slice(None)))
        u2_no_BC = fluid_context.U.get((1, slice(None), slice(None), slice(None)))
        u3_no_BC = fluid_context.U.get((2, slice(None), slice(None), slice(None)))

        #calculate fluid momentum
        if config.params.validate == True and rank==0:
            u1_mom_new=0
            u2_mom_new=0
            u3_mom_new=0
            for i in range(config.params.N[0]):
                for j in range(config.params.N[1]):
                    for k in range(config.params.N[2]):
                        u1_mom_new += u1_no_BC[i,j,k]
                        u2_mom_new += u2_no_BC[i,j,k]
                        u3_mom_new += u3_no_BC[i,j,k]
            u1_mom.append(u1_mom_new)
            u2_mom.append(u2_mom_new)
            u3_mom.append(u3_mom_new)

        #arrays are appended at the end with the upper boundary
        #periodic BC u(x_0)=u(x_n)
        #this is used to interpolate in the uppermost grid cells

        if rank==0:
            u1_BC_x=np.concatenate((u1_no_BC,u1_no_BC[:1,:,:]),axis=0)
            u1_BC_xy=np.concatenate((u1_BC_x,u1_BC_x[:,:1,:]), axis=1)
            u1=np.concatenate((u1_BC_xy,u1_BC_xy[:,:,:1]), axis=2)

            u2_BC_x=np.concatenate((u2_no_BC,u2_no_BC[:1,:,:]),axis=0)
            u2_BC_xy=np.concatenate((u2_BC_x,u2_BC_x[:,:1,:]), axis=1)
            u2=np.concatenate((u2_BC_xy,u2_BC_xy[:,:,:1]), axis=2)

            u3_BC_x=np.concatenate((u3_no_BC,u3_no_BC[:1,:,:]),axis=0)
            u3_BC_xy=np.concatenate((u3_BC_x,u3_BC_x[:,:1,:]), axis=1)
            u3=np.concatenate((u3_BC_xy,u3_BC_xy[:,:,:1]), axis=2)

        else:
            #initialese u1, u2, u3 on other ranks to be ready for broadcasting
            u1=None
            u2=None
            u3=None

        #broadcast array to all ranks

        u1=MPI.COMM_WORLD.bcast(u1,root=0)
        u2=MPI.COMM_WORLD.bcast(u2,root=0)
        u3=MPI.COMM_WORLD.bcast(u3,root=0)
        if rank == 0:
            print('2nd_2: Communication computed in {:.3f} sec'.format(time.time() - t_communication_0))

        #-------# Interpolate fluid velocities to mass point positions #-------#
        t_interpolate_0 = time.time()
        #coupleutils.interpolate_fibre(fiber_context, fluid_context)
        for i in range(fib_per_prc):
            coupleutils_2way.interpolate_fibre_v2(fiber_context[i], fluid_context, x1, x2, x3, u1, u2, u3)
        if rank == 0:
            print('2nd_3: Interpolation computed in {:.3f} sec'.format(time.time() - t_interpolate_0))
        #-------------------# Solve for the fiber dynamics #-------------------#

        # Call the iterative solver to get the solution at the next time step:
        t_iterative_0 = time.time()
        for i in range(fib_per_prc):
            fibutils_2way.advance_fibre(fiber_context[i], n, n_plus, dt_took, test, Fz_dyn)
        if rank == 0:
            print('3rd: Iterative computed in {:.3f} sec'.format(time.time() - t_iterative_0))

        # Update the variables for the new time step:
        t_updatefiber_0 = time.time()
        for i in range(fib_per_prc):
            fibutils_2way.update_fibre(fiber_context[i], n, n_plus, test, Fz_dyn)
            if config.params.validate == True:
                fibutils_2way.compute_fib_energy(fiber_context[i])
        if rank == 0:
            print('3rd_2: Update fiber computed in {:.3f} sec'.format(time.time() - t_updatefiber_0))
        # Interpolate fibre fources to fluid mesh for backcoupling

            #fibutils.check_variables(fiber_context)
        if config.params.tstep%config.params.write_fib==0:
                # I/O:
            if rank==0:
                t_IOfiber_0 = time.time()
            for i in range(fib_per_prc):
                coupleutils_2way.writetofile(fiber_context[i], stringfile[i])
            if rank == 0:
                print('3rd_3: IO fiber computed in {:.3f} sec'.format(time.time() - t_IOfiber_0))

            #if config.params.tstep % 50 == 0:
            #    coupleutils.check_variables(fiber_context
        if rank==0:
            t_backcoupling=time.time()
        # Interpolate fibre fources to fluid mesh for backcoupling
        Force_np=[] #store interpolated values
        for i in range(fib_per_prc):
            Force_np += coupleutils_2way.interpolate_backcoupling(fiber_context[i], fluid_context, nprocs, fib_per_prc=fib_per_prc, number=i+fib_num)
        Force=Array(fluid_context.VT) #shenfun object to allow fft
        coupleutils_2way.get_sourceterm_array(Force, Force_np, nprocs, rank, fib_per_prc, fib_num)
        fluid_context.Source=Force.forward() #fft
        """
        test fft
        u_f=coupleutils_2way.test_fft(fluid_context)
        fluid_context.Source = u_f.forward()
        coupleutils_2way.writesource(fluid_context.Source)
        """
        if config.params.validate == True:
            f1_new = Force.get((0, slice(None), slice(None), slice(None)))
            f2_new = Force.get((1, slice(None), slice(None), slice(None)))
            f3_new = Force.get((2, slice(None), slice(None), slice(None)))
            if rank==0:
                f1_add=0
                f2_add=0
                f3_add=0
                for i in range(config.params.N[0]):
                    for j in range(config.params.N[1]):
                        for k in range(config.params.N[2]):
                            f1_add += f1_new[i,j,k]
                            f2_add += f2_new[i,j,k]
                            f3_add += f3_new[i,j,k]
                f1.append(f1_add*config.params.dt)
                f2.append(f2_add*config.params.dt)
                f3.append(f3_add*config.params.dt)

        if fluid_solver.rank == 0:
        	print('3rd_4: Source term computed in {:.3f} sec'.format(time.time() - t_backcoupling))


        t_barrier_0 = time.time()
        fluid_solver.comm.Barrier()
        if rank == 0:
            print('4th: Barrier computed in {:.3f} sec'.format(time.time() - t_barrier_0))


        #str_animation_tstep = "NS_visualize/Isotropic_" + str(config.params.tstep) + "_fn" + str(config.params.fib_n) + "_N" + str(config.params.N) + "_dt" + str(config.params.dt) + "_tstep" + str(config.params.tstep) + ".png"
        #coupleutils.plot_isotropic_slice(fluid_context, fluid_solver, str_animation_tstep)

        #if config.params.tstep == 4000:
        #    fibutils.check_variables(fib_context)

        fluid_solver.timer()

        if not fluid_solver.profiler.getstats() and config.params.make_profile:
            #Enable profiling after first step is finished
            fluid_solver.profiler.enable()

        if fluid_solver.end_of_tstep(fluid_context):
            break
    config.params.dt = dt_in

    if config.params.make_profile:
        fluid_solver.results = fluid_solver.create_profile(fluid_solver.profiler)

    fluid_solver.regression_test(fluid_context)

    fluid_context.hdf5file.close()

    ###--------------------------- POSTPROCESSING ---------------------------###

    if fluid_solver.rank == 0:
        print("\n#------------------------ POSTPROCESSING ------------------------#\n")

    if config.params.validate==True and rank==0:

        coupleutils_2way.plot_momentum(u1_mom,u2_mom,u3_mom,f1,f2,f3, config.params.dt)
        coupleutils_2way.plot_energy(fiber_context,config.params.dt)
        coupleutils_2way.writemomentum(u1_mom,u2_mom,u3_mom,f1,f2,f3)

    coupleutils_2way.plot_isotropic_slice(fluid_context, fluid_solver, str_animation_final)

    fluid_solver.timer.final(config.params.verbose)

    if fib_per_prc >= 1:
        high_Re=max(fiber_context[0].Re)
        low_Re=min(fiber_context[0].Re)
        many_fev=max(fiber_context[0].nfev)
        few_fev=min(fiber_context[0].nfev)
        med_fev=statistics.median(fiber_context[0].nfev)
        if fib_per_prc>0 and fiber_context[0].convergence>0:
            convergence=[fib_num,fiber_context[0].convergence]
        else:
            convergence=[]
        for i in range(1,fib_per_prc):
            if max(fiber_context[i].Re) > high_Re:
                high_Re = max(fiber_context[i].Re)
            if low_Re > min(fiber_context[i].Re):
                low_Re = min(fiber_context[i].Re)
            if (max(fiber_context[i].nfev)) > many_fev:
                many_fev = max(fiber_context[i].nfev)
            if few_fev > min(fiber_context[i].nfev):
                few_fev = min(fiber_context[i].nfev)
            med_fev += statistics.median(fiber_context[i].nfev)
            if fiber_context[i].convergence>0:
                convergence.append(fib_num+i)
                convergence.append(fiber_context[i].convergence)
        if rank!=0:
            MPI.COMM_WORLD.send((high_Re, low_Re, many_fev, few_fev, med_fev, convergence), dest=0, tag=rank)
        else:
            max_Re = [high_Re]
            min_Re = [low_Re]
            max_fev = [many_fev]
            min_fev = [few_fev]
            for i in range(1,min(nprocs,config.params.fib_amt)):
                high, low, many, few, med, conv = MPI.COMM_WORLD.recv(source=i, tag=i)
                max_Re.append(high)
                min_Re.append(low)
                max_fev.append(many)
                min_fev.append(few)
                med_fev += med
                convergence+=conv



    if fluid_solver.rank == 0:
        print("\nReynolds number:\nmax(Re) =", max(max_Re), "\nmin(Re) =", min(min_Re))
        print("\nFunction evaluations:\nmax(nfev) =", max(max_fev), "\nmin(nfev) =", min(min_fev), "\nmedian(nfev) =", med_fev/config.params.fib_amt)
        for i in range(0,len(convergence),2):
            print("Fiber No "+str(convergence[i])+" dynamics did "+str(convergence[i+1])+" times not satisfy the equation of motion.")
        print("E", config.params.fib_E)
        print("A", config.params.fib_A)
        print("I", config.params.fib_I)
        print("eta", config.params.fib_eta)
        print("rho", config.params.fib_rho)
        print("nu", config.params.nu)
        print("kd", config.params.kd)
        print("elastic length", (config.params.fib_E*config.params.fib_I)**0.25/(config.params.rho*config.params.nu**2*config.params.kd**2)**0.25)
        print("fluid density", config.params.rho)
    """
    if fluid_solver.rank == 0:
        check_drag.check_drag_functions('normal', 'angles')
        check_drag.check_drag_functions('long', 'angles')
        check_drag.check_drag_functions('long', 'ratios')"""

    ###--------------------- PRINT WHICH JOB ---------------------###

    if fluid_solver.rank == 0:
        #print("\nJOB:\n mpirun -np 2 python FibreDNS.py --dt 0.001 --T 0.01 --fib_n 25 --fib_L 2 --init_from_file True --write_result 10 --checkpoint 10 --N 60 60 60 --plot_step 10 --optimization cython --integrator BS5_fixed NS\n")
        #print("\nJOB:\n bsub -n 24 mpirun python FibreDNS.py --dt 0.000078125 --T 0.1 --fib_n 20 --fib_L 3.2 --init_from_file True --write_result 1000 --checkpoint 10000 --N 128 128 128 --plot_step 1000 --optimization cython --integrator BS5_fixed NS")
        #print("\nJOB:\n bsub -n 24 mpirun python FibreDNS.py --dt 0.001 --T 0.01 --fib_n 20 --fib_L 3.2 --init_from_file True --write_result 1000 --checkpoint 10000 --N 256 256 256 --plot_step 1000 --optimization cython --integrator BS5_fixed NS")
        pass
