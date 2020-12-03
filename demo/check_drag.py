import matplotlib as mlb
import matplotlib.pyplot as plt
import numpy as np
import math
from cycler import cycler
from fibredyn import drag_coeffs
from spectralDNS import config

def check_drag_functions(which_comp, ang_or_rat_graph):

    which = which_comp          # 'long' or 'normal'
    ang_or_rat = ang_or_rat_graph   # 'angles' or 'ratios'

    if which == 'normal':
        angles = np.array([math.radians(15.), math.radians(30.),
                           math.radians(45.), math.radians(60.), math.radians(75.), math.radians(90.)])  # [º] - Angle of attack
    elif which == 'long':
        angles = np.array([math.radians(0.), math.radians(15.), math.radians(30.),
                           math.radians(45.), math.radians(60.), math.radians(75.)])  # [º] - Angle of attack

    ratio_L_to_d = np.array([10, 25, 100, 250, 1000, 2500, 10000])

    color1 = ['crimson', 'deepskyblue', 'darkorange', 'limegreen', 'darkviolet',
              'gold', 'green']
    color2 = ['lightsalmon', 'lightcoral', 'tomato', 'indianred', 'red', 'firebrick',
              'darkred']

    if which == 'normal':
        fig, ax = plt.subplots()
        #   plt.title('Normal Viscous Drag Coefficients vs. Reynolds number for alpha = 90º')
        ax.set_xlabel('Re')
        ax.set_ylabel('Drag coefficient')

    elif which == 'long':
        fig, ax = plt.subplots()
        #   plt.title('Normal Viscous Drag Coefficients vs. Reynolds number for alpha = 90º')
        ax.set_xlabel('Re')
        ax.set_ylabel('Drag coefficient')

    n = 0

    if which == 'long':
        if ang_or_rat == 'angles':
            how_many = 6
        elif ang_or_rat == 'ratios':
            how_many = 7
    else:
        how_many = 6

    # angles or angleslong
    for index in range(how_many):
        # for index in range(1):

        if which == 'normal':
            ratio = ratio_L_to_d[0]
            alpha = angles[index]
        elif which == 'long':
            if ang_or_rat == 'angles':
                ratio = ratio_L_to_d[4]
                alpha = angles[index]
            elif ang_or_rat == 'ratios':
                ratio = ratio_L_to_d[index]
                alpha = angles[0]

        ### ARRAYS ###

        power_Re_min = -2   # Re_min = 10^(-2)
        power_Re_max = 6    # Re_max = 10^(6)
        Re = np.logspace(power_Re_min, power_Re_max, num=1000)  # Re = rho * v_rel * d / mu
        Cn_hybrid = np.zeros(len(Re))
        Cl_hybrid = np.zeros(len(Re))

        for i in range(len(Re)):
            v_rel = config.params.nu * Re[i] / config.params.fib_d
            v_rel_ton = v_rel/math.sin(alpha) if math.sin(alpha) != 0 else 0
            v_rel_tol = v_rel/math.cos(alpha) if math.cos(alpha) != 0 else 0
            Cn_hybrid[i] = drag_coeffs.drag_coeff_normal(v_rel_ton, alpha)
            Cl_hybrid[i] = drag_coeffs.drag_coeff_long(v_rel_tol, alpha, ratio)

        ### PLOT ###

        if which == 'normal':
            ax.loglog(Re, Cn_hybrid, color=color1[n])
            #   ax.semilogx(Re, Cd_Kaplun)
            #   ax.semilogy(Re, Cd_Kaplun)
            #   ax.set_xlim(5*10**(-3), 2*10**(6))

        elif which == 'long':
            ax.loglog(Re, Cl_hybrid, color=color1[n])
            #   ax.semilogx(Re, Cd_Kaplun)
            #   ax.semilogy(Re, Cd_Kaplun)
            #   ax.set_xlim(5*10**(-3), 2*10**(6))

        n += 1

    if which == 'normal':
        #ax.set_ylim(10**(-1), 10**(3))
        plt.legend(['15º', '30º', '45º', '60º', '75º', '90º'])
        plt.xticks(np.logspace(-2, 6, num=9))
        fig.tight_layout()
        ax.grid()
        plt.gca().set_aspect('equal')

    elif which == 'long':
        #ax.set_ylim(10**(-1), 10**(3))
        if ang_or_rat == 'angles':
            plt.legend(['0º', '15º', '30º', '45º', '60º', '75º'])
        elif ang_or_rat == 'ratios':
            plt.legend([r'$L /d = 10$', r'$L /d = 25$', r'$L /d = 100$',
                        r'$L /d = 250$', r'$L /d = 1000$', r'$L /d = 2500$', r'$L /d = 10000$'])
        plt.xticks(np.logspace(-2, 6, num=9))
        fig.tight_layout()
        ax.grid()
        plt.gca().set_aspect('equal')

    # List of named colors: https://matplotlib.org/gallery/color/named_colors.html
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    plt.rcParams['font.variant'] = 'small-caps'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.linestyle'] = '-'

    if which == 'normal':
        fig.savefig("Check_Solver/Cn_hybrid_angles.png")

    elif which == 'long':
        if ang_or_rat == 'angles':
            fig.savefig("Check_Solver/Cl_hybrid_angles.png")
        elif ang_or_rat == 'ratios':
            fig.savefig("Check_Solver/Cl_hybrid_ratios.png")
