from numba import jit

@jit(nopython=True, fastmath=True)
def loop1(U_hat, U_hat0, U_hat1):
    for i in range(U_hat.shape[0]):
        for j in range(U_hat.shape[1]):
            for k in range(U_hat.shape[2]):
                for l in range(U_hat.shape[3]):
                    z = U_hat[i, j, k, l]
                    U_hat1[i, j, k, l] = z
                    U_hat0[i, j, k, l] = z

@jit(nopython=True, fastmath=True)
def loop2(dU, U_hat, U_hat0, b, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat[i, j, k, l] = U_hat0[i, j, k, l] + b*dt*dU[i, j, k, l]

@jit(nopython=True, fastmath=True)
def loop3(dU, U_hat1, a, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat1[i, j, k, l] = U_hat1[i, j, k, l] + a*dt*dU[i, j, k, l]

@jit(nopython=True, fastmath=True)
def loop4(U_hat, U_hat1):
    for i in range(U_hat.shape[0]):
        for j in range(U_hat.shape[1]):
            for k in range(U_hat.shape[2]):
                for l in range(U_hat.shape[3]):
                    U_hat[i, j, k, l] = U_hat1[i, j, k, l]

def RK4(U_hat, U_hat0, U_hat1, dU, a, b, dt, solver, context):
    loop1(U_hat, U_hat0, U_hat1)
    c = context
    for rk in range(4):
        dU = solver.ComputeRHS(dU, U_hat, solver, **c)
        if rk < 3:
            loop2(dU, U_hat, U_hat0, b[rk], dt)
        loop3(dU, U_hat1, a[rk], dt)
    loop4(U_hat, U_hat1)
    return U_hat, dt, dt

@jit(nopython=True, fastmath=True)
def loop5(dU, U_hat, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat[i, j, k, l] += dU[i, j, k, l]*dt

@jit(nopython=True, fastmath=True)
def loop6(dU, U_hat, U_hat0, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat[i, j, k, l] = U_hat[i, j, k, l] + 1.5*dU[i, j, k, l]*dt - 0.5*U_hat0[i, j, k, l]

@jit(nopython=True, fastmath=True)
def loop7(dU, U_hat0, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat0[i, j, k, l] = dU[i, j, k, l]*dt

def ForwardEuler(U_hat, dU, dt, solver, context):
    dU = solver.ComputeRHS(dU, U_hat, solver, **context)
    loop5(dU, U_hat, dt)
    return U_hat, dt, dt

def AB2(U_hat, U_hat0, dU, dt, tstep, solver, context):
    dU = solver.ComputeRHS(dU, U_hat, solver, **context)
    if tstep == 0:
        loop5(dU, U_hat, dt)
    else:
        loop6(dU, U_hat, U_hat0, dt)
    loop7(dU, U_hat0, dt)
    return U_hat, dt, dt

@jit(nopython=True, fastmath=True)
def cross1(c, a, b):
    """Regular c = a x b"""
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            for k in range(a.shape[3]):
                a0 = a[0, i, j, k]
                a1 = a[1, i, j, k]
                a2 = a[2, i, j, k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = a1*b2 - a2*b1
                c[1, i, j, k] = a2*b0 - a0*b2
                c[2, i, j, k] = a0*b1 - a1*b0
    return c

@jit(nopython=True, fastmath=True)
def cross2a(c, a, b):
    """ c = 1j*(a x b)"""
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            for k in range(a.shape[3]):
                a0 = a[0, i, j, k]
                a1 = a[1, i, j, k]
                a2 = a[2, i, j, k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = -(a1*b2.imag - a2*b1.imag) + 1j*(a1*b2.real - a2*b1.real)
                c[1, i, j, k] = -(a2*b0.imag - a0*b2.imag) + 1j*(a2*b0.real - a0*b2.real)
                c[2, i, j, k] = -(a0*b1.imag - a1*b0.imag) + 1j*(a0*b1.real - a1*b0.real)
    return c

@jit(nopython=True, fastmath=True)
def cross2c(c, a0, a1, a2, b):
    """ c = 1j*(a x b)"""
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            for k in range(b.shape[3]):
                a00 = a0[i, 0, 0]
                a11 = a1[0, j, 0]
                a22 = a2[0, 0, k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = -(a11*b2.imag - a22*b1.imag) + 1j*(a11*b2.real - a22*b1.real)
                c[1, i, j, k] = -(a22*b0.imag - a00*b2.imag) + 1j*(a22*b0.real - a00*b2.real)
                c[2, i, j, k] = -(a00*b1.imag - a11*b0.imag) + 1j*(a00*b1.real - a11*b0.real)
    return c

def cross2(c, a, b):
    if isinstance(a, list):
        c = cross2c(c, a[0], a[1], a[2], b)
    else:
        c = cross2a(c, a, b)
    return c

def add_pressure_diffusion_NS(du, u_hat, nu, ksq, kk, p_hat, k_over_k2):
    du = add_pressure_diffusion_NS_(du, u_hat, nu, ksq, kk[0][:, 0, 0],
                                    kk[1][0, :, 0], kk[2][0, 0, :], p_hat, k_over_k2)
    return du

@jit(nopython=True, fastmath=True)
def add_pressure_diffusion_NS_(du, u_hat, nu, ksq, kx, ky, kz, p_hat, k_over_k2):
    for i in range(ksq.shape[0]):
        k0 = kx[i]
        for j in range(ksq.shape[1]):
            k1 = ky[j]
            for k in range(ksq.shape[2]):
                z = nu*ksq[i, j, k]
                k2 = kz[k]
                p_hat[i, j, k] = du[0, i, j, k]*k_over_k2[0, i, j, k]+du[1, i, j, k]*k_over_k2[1, i, j, k]+du[2, i, j, k]*k_over_k2[2, i, j, k]
                du[0, i, j, k] = du[0, i, j, k] - (p_hat[i, j, k]*k0+u_hat[0, i, j, k]*z)
                du[1, i, j, k] = du[1, i, j, k] - (p_hat[i, j, k]*k1+u_hat[1, i, j, k]*z)
                du[2, i, j, k] = du[2, i, j, k] - (p_hat[i, j, k]*k2+u_hat[2, i, j, k]*z)
    return du

@jit(nopython=True, fastmath=True)
def compute_vw(u_hat, f_hat, g_hat, k_over_k2):
    for i in range(u_hat.shape[1]):
        for j in range(u_hat.shape[2]):
            for k in range(u_hat.shape[3]):
                u_hat[1, i, j, k] = -1j*(k_over_k2[0, i, j, k]*f_hat[i, j, k] - k_over_k2[1, i, j, k]*g_hat[i, j, k])
                u_hat[2, i, j, k] = -1j*(k_over_k2[1, i, j, k]*f_hat[i, j, k] + k_over_k2[0, i, j, k]*g_hat[i, j, k])
    return u_hat

def mult_K1j(K, a, f):
    f = _mult_K1j(K[1][0, :, 0], K[2][0, 0], a, f)
    return f

@jit(nopython=True, fastmath=True)
def _mult_K1j(Ky, Kz, a, f):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                f[0, i, j, k] = 1j*Kz[k]*a[i, j, k]
                f[1, i, j, k] = -1j*Ky[j]*a[i, j, k]
    return f
