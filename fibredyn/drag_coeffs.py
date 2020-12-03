""" Drag coefficients to calculate the drag forces
    according to the drag cylinder model defined """

import math
from spectralDNS import config

Euler_cte = 0.5772156649  # Euler's constant
Re_crit_1_n = 0.1    # Critical value of Re for the definition of Cd (Kaplun)
Re_crit_1_l = 0.1    # Critical value of Re for the definition of Cd (Kaplun)
Re_crit_2_n = 10.
Re_crit_2_l = 10.

def func_Re(vel):
    return config.params.fib_d * vel / config.params.nu

def func_Eps(Re):
    # epsilon parameter for the Kaplun drag coefficient
    return (0.5 - Euler_cte - math.log(Re / 8.))**(-1)

def func_Cn_Kaplun(Re, alpha):
    #total normal drag coef. (Kaplun)
    return (8. * math.pi / Re) * func_Eps(Re) * (1. - 0.87 *
                                    (func_Eps(Re))**2) * (math.sin(alpha))**2

def func_Cn_Munson(Re, alpha):
    #total normal drag coef. (Munson)
    return (5.93 / math.sqrt(Re) + 1.17) * (math.sin(alpha))**2

def func_Cl_KellerRubinow(Re, alpha, ratio):
    # total longitudinal drag coef. (Keller-Rubinow)
    # I MODIFIED math.cos(alpha) -> math.cos(alpha)**2 (check!)
    return (4 * math.pi / Re) * math.cos(alpha)**2 * \
        (math.log(2. * ratio) - 3. / 2. + math.log(2) - (1 - math.pi**2 / 12.) /
         (math.log(2. * ratio)))**(-1)   # total longitudinal drag coef. (Keller-Rubinow)

def func_Cl_f(Re, alpha):
    # total longitudinal drag coef. (Thom)
    #return (math.cos(alpha))**(3./2.) / math.sqrt(Re)
    return (math.cos(alpha))**(2.) * 4. / math.sqrt(Re)

def func_a_n(Re, alpha):
    return math.log10(func_Cn_Munson(Re_crit_2_n, alpha) /
                     func_Cn_Kaplun(Re_crit_1_n, alpha)) / math.log10(100)

def func_b_n(Re, alpha):
    return math.log10(func_Cn_Munson(Re_crit_2_n, alpha) / (10**func_a_n(Re, alpha)))

def func_a_l(Re, alpha, ratio):
    return math.log10(func_Cl_f(Re_crit_2_l, alpha) /
                     func_Cl_KellerRubinow(Re_crit_1_l, alpha, ratio)) / math.log10(100)

def func_b_l(Re, alpha, ratio):
    return math.log10(func_Cl_f(Re_crit_2_l, alpha) / (10**func_a_l(Re, alpha, ratio)))

def func_interpolate_n(Re, alpha):
    # total normal drag coefficient interpolation
    return (Re**(func_a_n(Re, alpha))) * (10**(func_b_n(Re, alpha)))

def func_interpolate_l(Re, alpha, ratio):
    # total longitudinal drag coefficient interpolation
    return (Re**(func_a_l(Re, alpha, ratio))) * (10**(func_b_l(Re, alpha, ratio)))


### Calculate the drag coefficients ###

def drag_coeff_normal(v_rel, alpha):
    # need angle in radians!
    Re_normal = func_Re(v_rel * math.sin(alpha))
    if Re_normal != 0.:
        if Re_normal < 0.1:
            C_n = func_Cn_Kaplun(Re_normal, alpha)
        elif Re_normal >= 0.1 and Re_normal < 10.:
            C_n = func_interpolate_n(Re_normal, alpha)
        elif Re_normal >= 10.:
            C_n = func_Cn_Munson(Re_normal, alpha)
        return C_n
    else:
        return 0.


def drag_coeff_long(v_rel, alpha, ratio):
    # need angle in radians!
    Re_long = func_Re(v_rel * math.cos(alpha))
    if Re_long != 0.:
        if Re_long < 0.1:
            C_l = func_Cl_KellerRubinow(Re_long, alpha, ratio)
        elif Re_long >= 0.1 and Re_long < 10.:
            C_l = func_interpolate_l(Re_long, alpha, ratio)
        elif Re_long >= 10.:
            C_l = func_Cl_f(Re_long, alpha)
        return C_l
    else:
        return 0.
