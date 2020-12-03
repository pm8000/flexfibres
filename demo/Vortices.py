import matplotlib.pyplot as plt
from numpy import random, exp, pi
from spectralDNS import config, get_solver, solve

def initialize(curl, W_hat, T, X, **context):
    curl[:] = 0
    if config.params.init == 'random':
        curl[:] = 0.3*random.randn(*curl.shape)

    else:
        curl[0] = exp(-((X[1]-pi)**2+(X[2]-pi+pi/4)**2)/(0.2)) \
             + exp(-((X[1]-pi)**2+(X[2]-pi-pi/4)**2)/(0.2)) \
             - 0.5*exp(-((X[1]-pi-pi/4)**2+(X[2]-pi-pi/4)**2)/(0.4))

    W_hat = curl.forward(W_hat)

def set_source(U, Source, X, **context):
    U[:] = 0
    if config.params.init == 'random':
        U[0] = 100*exp(-((X[1]-pi)**2+(X[2]-pi+pi/4)**2)/0.01) \
            + 100*exp(-((X[1]-pi)**2+(X[2]-pi-pi/4)**2)/0.01) \
            - 0.5*100*exp(-((X[1]-pi-pi/4)**2+(X[2]-pi-pi/4)**2)/0.04)
    else:
        pass

    Source = U.forward(Source)

im, im2 = None, None
def update(context):
    global im, im2
    c = context
    params = config.params
    solver = config.solver
    X, U = c.X, c.U

    if (params.tstep % params.plot_step == 0 and params.plot_step > 0 or
            params.tstep == 1):
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)

    if params.tstep == 1 and solver.rank == 0:
        plt.figure()
        im = plt.quiver(X[1][0, :, 0], X[2][0, 0],
                        U[2, 0], U[1, 0],
                        pivot='mid', scale=2)

        plt.figure()
        im2 = plt.imshow(curl[0, 0, ::-1, :])
        plt.draw()

        plt.pause(1e-6)
        globals().update(im=im, im2=im2)

    if params.tstep == 3:
        c.Source[:] = 0

    if params.tstep % params.plot_step == 0 and solver.rank == 0:
        im.set_UVC(U[2, 0], U[1, 0])
        im2.set_data(curl[0, 0, ::-1, :])
        im2.autoscale()
        plt.pause(1e-6)

    print("Time = ", params.t)

def regression_test(context):
    global im
    solver = config.solver
    U = solver.get_velocity(**context)
    im.set_UVC(U[1, :, :, 0], U[2, :, :, 0])
    plt.pause(1e-6)

if __name__ == "__main__":
    config.update(
        {'nu': 0.000625,              # Viscosity
         'dt': 0.01,                  # Time step
         'T': 50,                     # End time
         'write_result': 10
        }, 'triplyperiodic')
    config.triplyperiodic.add_argument("--init", default='random', choices=('random', 'vortex'))
    config.triplyperiodic.add_argument("--plot_step", type=int, default=10) # required to allow overloading through commandline
    solver = get_solver(update=update,
                        regression_test=regression_test,
                        mesh="triplyperiodic")

    assert config.params.solver == 'VV'
    assert config.params.decomposition == 'slab'
    context = solver.get_context()
    initialize(**context)
    set_source(**context)
    solve(solver, context)
