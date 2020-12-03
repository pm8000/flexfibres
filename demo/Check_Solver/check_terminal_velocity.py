from scipy import optimize
import math

def compute_terminal_velocity(rho_p, d):
    g = 9.81
    rho_f = 1.299
    nu_f = 0.005429
    mp = 0.018
    dp = (24*mp/(4*math.pi*rho_p))**(1/3)   # diameter particle (spider)
    print(d, dp)
    tau_St = dp**2 * rho_p / (18 * rho_f * nu_f)
    def fun(vp):
        f_vp = g*(rho_p-rho_f)/rho_p - vp*(1+0.15*(dp*vp/nu_f)**0.687)/tau_St
        return f_vp
    def funprime(vp):
        fprime_vp = -1/tau_St - 0.15*vp**0.687 *(dp/nu_f)**0.687 /tau_St
        return fprime_vp
    result = optimize.root_scalar(fun, x0=0.1, fprime=funprime, method="newton", xtol=1e-12)
    return result.root, result.iterations, result.function_calls

print(compute_terminal_velocity(1320, 231e-06))
