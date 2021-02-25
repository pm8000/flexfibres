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
from FibreSolver import get_fibcontext, initialize_fibre
from fibredyn import fibutils
from fibreHIT import coupleutils, coupleutils_2way
import numpy as np
import h5py     # Reference: http://docs.h5py.org/en/stable/
#import pdb     # -> Python Debugger
import check_drag

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
    config.triplyperiodic.add_argument("--no_spider", type=bool, default=False,
        help="have fibres with free ends")

    ###---------------- FLUID SOLVER, ADDITIONAL PARAMETERS ----------------###

    # Passing the function 'update()' defined above (in Isotropic.py):
    fluid_solver = get_solver(update=update, mesh="triplyperiodic")  # see __init__.py

    # Additional flow parameters:
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

    # Choose initial orientation of the fiber. The fiber is initialized by
    # specifying the location of the 0^th mass point and the axis along which
    # it extends:
    if config.params.termvel:
        config.params.fib_start = [np.pi, np.pi, 2*np.pi/4]
        config.params.fib_axis = 3
    else:
        config.params.fib_start = [2*np.pi/4, np.pi, np.pi]
        config.params.fib_axis = 1

    # Get the fluid and fiber contexts to solve the fibre dynamics:
    fluid_context = fluid_solver.get_context()
    fiber_context = get_fibcontext()

    # initialize fluid flow and fiber dynamics:
    initialize(fluid_solver, fluid_context)
    if config.params.termvel == False:
        initialize_fibre(fiber_context, config.params.fib_start, config.params.fib_axis)
    else:
        terminal_velocity, iterations, function_calls = coupleutils.compute_terminal_velocity()
        axis_spider = 3
        if fluid_solver.rank == 0:
            print("\nTerminal velocity by the Newton method: {} m/s\nIterations: {}\nFunction_calls: {}\n".format(terminal_velocity, iterations, function_calls))
        terminal_velocity = - terminal_velocity
        initialize_fibre(fiber_context, config.params.fib_start, config.params.fib_axis,
                        terminal_velocity, axis_spider)


    # NOTE: The contexts are dictionaries of the class AttributeDict

    # Check the values of all the parameters defined for the fiber dynamics:
    if fluid_solver.rank == 0:
        coupleutils.check_parameters(fluid_solver)

    # HDF5 file name:
    #filestr = "Results/NS_coupled_isoDNS_{}_{}_{}_T{}"
    #fluid_context.hdf5file.filename = filestr.format(*config.params.N, config.params.T)
    #filestr = "Results/NS_coupled_isoDNS_{}_{}_{}_dt{}"
    #fluid_context.hdf5file.filename = filestr.format(*config.params.N, config.params.dt)
    filestr = "Results/NS_coupled_isoDNS_{}_{}_{}_fibn{}_dt{}_T{}"
    fluid_context.hdf5file.filename = filestr.format(*config.params.N, config.params.fib_n, config.params.dt, config.params.T)

    # Fibre I/O:
    if config.params.termvel:
        stringfile1 = "./Results/fiber_x1_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + "_TV.csv"
        stringfile2 = "./Results/fiber_x2_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + "_TV.csv"
        stringfile3 = "./Results/fiber_x3_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + "_TV.csv"
        stringfile4 = "./Results/force_x1_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + "_TV.csv"
        stringfile5 = "./Results/force_x2_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + "_TV.csv"
        stringfile6 = "./Results/force_x3_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + "_TV.csv"
    else:
        stringfile1 = "./Results/fiber_x1_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + ".csv"
        stringfile2 = "./Results/fiber_x2_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + ".csv"
        stringfile3 = "./Results/fiber_x3_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + ".csv"
        stringfile4 = "./Results/force_x1_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + ".csv"
        stringfile5 = "./Results/force_x2_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + ".csv"
        stringfile6 = "./Results/force_x3_fn" + str(config.params.fib_n) + "_N" + str(config.params.N[0]) + "_dt" + str(config.params.dt) + "_T" + str(config.params.T) + ".csv"

    # Isotropic plot:
    str_animation_final = "NS_visualize/Isotropic_" + str(config.params.tstep) + "_fn" + str(config.params.fib_n) + "_N" + str(config.params.N) + "_dt" + str(config.params.dt) + ".png"
    str_animation_init = "NS_visualize/Isotropic_init_" + str(config.params.tstep) + "_fn" + str(config.params.fib_n) + "_N" + str(config.params.N) + "_dt" + str(config.params.dt) + ".png"

    ### Not called at the beggining! Called afterwards when willing to use the
    ### file as initial condition (previous simulation required):
    if config.params.init_from_file:
        fileinitstr = "NS_isotropic_{}_{}_{}".format(*config.params.N) + "_c.h5"
        init_from_file(fileinitstr, fluid_solver, fluid_context)

    # Data for the 'NS_coupled_isoDNS_{}_{}_{}_T{}.h5' file:
    Ek, bins, E0, E1, E2 = spectrum(fluid_solver, fluid_context)
    fluid_context.spectrumname = fluid_context.hdf5file.filename+".h5"
    f = h5py.File(fluid_context.spectrumname, mode='w', driver='mpio', comm=fluid_solver.comm)
    f.create_group("Turbulence")
    f["Turbulence"].create_group("Ek")
    bins = np.array(bins)
    f["Turbulence"].create_dataset("bins", data=bins)
    f.close()

    ### PRINT ###
    if fluid_solver.rank == 0:
        #print(int(config.params.T * 1 / config.params.dt))
        pass

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
    #changed by Pascal Müller to use dx for interpolation

    dx = 2*np.pi / config.params.N[0]
    L_1 = 2*np.pi - dx
    x1 = np.linspace(0, L_1, config.params.N[0])
    x2 = np.linspace(0, L_1, config.params.N[1])
    x3 = np.linspace(0, L_1, config.params.N[2])
    if fluid_solver.rank==0:
    	print("Grid spacing: ",dx)
    print("Rank", rank)

    while config.params.t + config.params.dt <= config.params.T+1e-12:

        t_fluid_0 = time.time()

        u, config.params.dt, dt_took = integrate()

        config.params.t += dt_took
        config.params.tstep += 1

        # Print to track the evolution of the simulation:
        if fluid_solver.rank == 0:
            print("\nTime step: ", config.params.tstep)
            b=False

        #---------------------# Solve for the fluid flow #---------------------#
        # Solve fluid flow:
        fluid_solver.update(fluid_context, backcoupling=True)

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
        fiber_context.u1 = fluid_context.U.get((0, slice(None), slice(None), slice(None)))
        fiber_context.u2 = fluid_context.U.get((1, slice(None), slice(None), slice(None)))
        fiber_context.u3 = fluid_context.U.get((2, slice(None), slice(None), slice(None)))
        if rank == 0:
            print('2nd_2: Communication computed in {:.3f} sec'.format(time.time() - t_communication_0))

        #uprint = fluid_context.U.get((0, 15, 2, 9))
        #print(fluid_context.U.global_shape) # -> (3, N[0], N[1], N[2])

        #print("Rank", fluid_solver.rank,"step", config.params.tstep, fluid_context.VT.shape())
        #print("Rank", fluid_solver.rank,"step", config.params.tstep, fluid_context.Source.shape)

        if fluid_solver.rank == 0:

            #print("U[0, 19, 19, 19]", uprint)  # only the rank that has it prints!!
            #print("u1:", np.shape(fiber_context.u1), type(fiber_context.u1)) # -> (N[0], N[1], N[2])
            #print("u2:", np.shape(fiber_context.u2), type(fiber_context.u2)) # -> (N[0], N[1], N[2])
            #print("u3:", np.shape(fiber_context.u3), type(fiber_context.u3)) # -> (N[0], N[1], N[2])
            #print("u1:", fiber_context.u1[0, 0, 0])
            #print("u2:", fiber_context.u2[0, 0, 0])
            #print("u3:", fiber_context.u3[0, 0, 0])

            #-------# Interpolate fluid velocities to mass point positions #-------#
            t_interpolate_0 = time.time()
            #coupleutils.interpolate_fibre(fiber_context, fluid_context)
            coupleutils.interpolate_fibre_v2(fiber_context, fluid_context, x1, x2, x3,
                            fiber_context.u1, fiber_context.u2, fiber_context.u3)
            if rank == 0:
                print('2nd_3: Interpolation computed in {:.3f} sec'.format(time.time() - t_interpolate_0))

            #-------------------# Solve for the fiber dynamics #-------------------#

            # Call the iterative solver to get the solution at the next time step:
            t_iterative_0 = time.time()
            fibutils.advance_fibre(fiber_context, n, n_plus, dt_took, test, Fz_dyn)
            if rank == 0:
                print('3rd: Iterative computed in {:.3f} sec'.format(time.time() - t_iterative_0))

            # Update the variables for the new time step:
            t_updatefiber_0 = time.time()
            fibutils.update_fibre(fiber_context, n, n_plus, test, Fz_dyn)
            if rank == 0:
                print('3rd_2: Update fiber computed in {:.3f} sec'.format(time.time() - t_updatefiber_0))
        # Interpolate fibre fources to fluid mesh for backcoupling

            #fibutils.check_variables(fiber_context)
            # I/O:
            if rank==0:
                t_IOfiber_0 = time.time()
            coupleutils.writetofile(fiber_context, stringfile1, stringfile2, stringfile3)
            if rank == 0:
                print('3: IO fiber computed in {:.3f} sec'.format(time.time() - t_IOfiber_0))

            #coupleutils_2way.writeforce(Force, stringfile4, stringfile5,stringfile6)
            #coupleutils_2way.writesource(fluid_context.Source)

            #if config.params.tstep % 50 == 0:
            #    coupleutils.check_variables(fiber_context
            if rank==0:
                t_backcoupling=time.time()
            # Interpolate fibre fources to fluid mesh for backcoupling
                Force_np=coupleutils_2way.interpolate_backcoupling(fiber_context, fluid_context,dx, nprocs)
        else:
            Force_np=coupleutils_2way.get_sourceterm_array(rank, nprocs)
            #Simulate a sinusoidal Force input to check for proper fourier transform
            #Force=coupleutils_2way.test_fft(fluid_context), only suitable for single thread
        coupleutils_2way.add_backcoupling(fluid_context,Force_np)
        if fluid_solver.rank == 0:
        	print('3rd_4: Source term computed in {:.3f} sec'.format(time.time() - t_updatefiber_0))


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

    coupleutils.plot_isotropic_slice(fluid_context, fluid_solver, str_animation_final)

    fluid_solver.timer.final(config.params.verbose)

    if fluid_solver.rank == 0:
        print("\nReynolds number:\nmax(Re) =", max(fiber_context.Re), "\nmin(Re) =", min(fiber_context.Re))
        print("\nFunction evaluations:\nmax(nfev) =", max(fiber_context.nfev), "\nmin(nfev) =", min(fiber_context.nfev), "\nmedian(nfev) =", statistics.median(fiber_context.nfev))

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
