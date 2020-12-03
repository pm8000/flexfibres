from scipy import optimize
from fibredyn.drag_coeffs import func_Cl_KellerRubinow, func_interpolate_l, func_Cl_f
import math

def compute_terminal_velocity(rho_p, d):
    g = 9.81
    #rho_f = 998
    #nu_f = 9.6e-07
    rho_f = 1.299
    nu_f = 0.005429
    #nu_f = 1.81e-05
    mp = 0.018
    dp = (24*mp/(4*math.pi*rho_p))**(1/3)   # diameter particle (spider)
    tau_St = dp**2 * rho_p / (18 * rho_f * nu_f)
    print(d, dp)
    L = 3.2
    def fun(vp):
        Re = d*vp/nu_f
        if Re < 0.1:
            #print("KR")
            C_l = func_Cl_KellerRubinow(Re, 0, L/d)
        elif Re >= 0.1 and Re < 10.:
            #print("Interp")
            C_l = func_interpolate_l(Re, 0, L/d)
        elif Re >= 10.:
            #print("Thom")
            C_l = func_Cl_f(Re, 0)
        f_vp = g*(rho_p-rho_f)/rho_p - vp*(1+0.15*(dp*vp/nu_f)**0.687)/tau_St - 2*rho_f*C_l*vp**2 / (math.pi*d*rho_p)
        return f_vp
    def funprime(vp):
        Re = d*vp/nu_f
        if Re < 0.1:
            #print("KR")
            C_l = func_Cl_KellerRubinow(Re, 0, L/d)
        elif Re >= 0.1 and Re < 10.:
            #print("Interp")
            C_l = func_interpolate_l(Re, 0, L/d)
        elif Re >= 10.:
            #print("Thom")
            C_l = func_Cl_f(Re, 0)
        fprime_vp = -1/tau_St - 0.15*vp**0.687 *(dp/nu_f)**0.687 /tau_St - 4*rho_f*C_l*vp / (math.pi*d*rho_p)
        return fprime_vp
    result = optimize.root_scalar(fun, x0=0.1, fprime=funprime, method="newton", xtol=1e-12)
    return result.root, result.iterations, result.function_calls

#nu_f = 9.6e-07
nu_f = 0.005429
print(compute_terminal_velocity(1320, 231e-06), "- Re:", compute_terminal_velocity(1320, 231e-06)[0]*231e-06/nu_f)
