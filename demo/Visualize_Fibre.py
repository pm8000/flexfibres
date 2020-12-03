import numpy as np
import csv
import matplotlib.pyplot as plt


def visualize_linstat():
    x = []
    time = []
    stringfile = "./csv_files/visualization_lin_" + str(fib_n) + ".csv"
    with open(stringfile, newline='') as csvfile:
        visualreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in visualreader:
            x.append(abs(float(row[0])))
            time.append(float(row[1]))

    k = fib_E * fib_A / fib_L
    m = fib_m0 + fib_rho * fib_A * fib_L / 3
    F = abs(fib_BC_Fz) + m * g
    x_mean = F / k
    w_0 = np.sqrt(k / m)
    print("w_0 =", w_0)
    c_damp = fib_eta * fib_A / fib_L
    zeta = c_damp / (2. * m)
    w_d = np.sqrt(w_0**2 - zeta**2)
    x_mean_l = np.array(len(x) * [x_mean])
    x_plus = np.array([x_mean*(1.+np.exp(-zeta*t)) for t in time])
    x_minus = np.array([x_mean*(1.-np.exp(-zeta*t)) for t in time])

    def local_maxima(x1, x2, x3):
        if x1 < x2 and x3 < x2:
            return True
        else:
            return False

    def local_minima(x1, x2, x3):
        if x1 > x2 and x3 > x2:
            return True
        else:
            return False

    n_max = 0
    n_min = 0
    maxima = []
    minima = []
    for i in range(1, len(x)-1):
        x_previous = x[i-1]
        x_current = x[i]
        x_next = x[i+1]
        if local_maxima(x_previous, x_current, x_next):
            n_max += 1
            maxima.append([x_current, time[i]])
        elif local_minima(x_previous, x_current, x_next):
            n_min += 1
            minima.append([x_current, time[i]])

    error_plus = 0
    error_minus = 0
    for i in range(n_max):
        error_plus += ((maxima[i][0] - x_mean*(1.+np.exp(-zeta*maxima[i][1]))) / x_mean)**2
    for i in range(n_min):
        error_minus += ((minima[i][0] - x_mean*(1.-np.exp(-zeta*minima[i][1]))) / x_mean)**2

    error_rms_x_plus = np.sqrt((1. / n_max) * error_plus)
    error_rms_x_minus = np.sqrt((1. / n_min) * error_minus)

    periods = []
    for i in range(1, n_min):
        t = minima[i][1] - minima[i-1][1]
        periods.append(t)
    period_sim = sum(periods) / len(periods)
    w_d_sim = 2 * np.pi / period_sim
    error_w_d = (w_d_sim - w_d) / w_d

    print("###################### ERRORS ######################\n")
    print("n_max = ", n_max)
    print("n_min = ", n_min)
    print("w_d = ", w_d, "w_d_sim = ", w_d_sim)
    print("n = ", fib_n,": error_rms_x_plus = ", error_rms_x_plus)
    print("n = ", fib_n,": error_rms_x_minus = ", error_rms_x_minus)
    print("n = ", fib_n,": error_w_d = ", error_w_d)
    print("Underdamped harmonic oscillator (c < 2*sqrt(k·m)): ", c_damp, " < ", 2*np.sqrt(k*m))

    fig, ax = plt.subplots()
    #plt.title('Spider position')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')
    plt.plot(time, x, color='b')
    plt.plot(time, x_plus, color='g', linestyle='dashed')
    plt.plot(time, x_minus, color='r', linestyle='dashed')
    plt.plot(time, x_mean_l, color='k', linestyle='dashed')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.rcParams['lines.linewidth'] = 0.1
    string = "./png_files/spider_position_lin_" + str(fib_n) + ".png"
    fig.savefig(string)


def visualize_lindyn():
    x = []
    x0 = []
    x1 = []
    time = []
    Fz = []
    #stringfile = "./csv_files/visualization_dyn_" + str(int(fib_eta/1000)) + "_" + str(fib_w_forced_ratio) + ".csv"
    stringfile = "./csv_files/visualization_dyn_" + str(int(fib_eta/1000)) + "_" + str(fib_w_forced_ratio) + ".csv"
    with open(stringfile, newline='') as csvfile:
        visualreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in visualreader:
            x.append(float(row[0]))
            x0.append(float(row[4]))
            x1.append(float(row[3]))
            time.append(float(row[1]))
            Fz.append(float(row[2]) * 10**(-7))

    k = fib_E * fib_A / fib_L
    m = fib_m0 + fib_rho * fib_A * fib_L / 3
    F = abs(fib_BC_Fz) + m * g
    F0 = abs(fib_BC_Fz)
    w_0 = np.sqrt(k / m)
    print("w_0 =", w_0)
    print("w_forced =", fib_w_forced)
    c_damp = fib_eta * fib_A / fib_L
    zeta = c_damp / (2. * np.sqrt(k * m))
    r = fib_w_forced / w_0    # resonance at r = 1 approx.
    #amp = (F0 / k) * ((1 - r**2)**2 + (2 * zeta * r)**2)**(-1)
    amp = F0 / np.sqrt(m**2 * (fib_w_forced**2 - w_0**2)**2 + c_damp**2 * fib_w_forced**2)
    if r == 1.0:
        ph_sh = -np.pi / 2
    elif r == 0.0:
        ph_sh = 0.
    else:
        ph_sh = np.arctan(-2 * zeta * r / (1 - r**2))
        if ph_sh > 0.:
            ph_sh -= np.pi

    def local_maxima(x1, x2, x3):
        if x1 < x2 and x3 < x2:
            return True
        else:
            return False

    def local_minima(x1, x2, x3):
        if x1 > x2 and x3 > x2:
            return True
        else:
            return False

    if r == 0.0:
        amp_sim = abs(x[-1])
        print("amp =", amp)
        print("amp_sim =", amp_sim)
        ph_sh_sim = 0.
        print("ph_sh =", ph_sh)
        print("ph_sh_sim =", ph_sh_sim)
    else:
        # For 500 0.9:
        x_slice = x[175000:225001]
        F_slice = Fz[175000:225001]
        time_slice = time[175000:225001]
        """
        if fib_eta == 500.0e3:
            x_slice = x[80000:90001]
            F_slice = Fz[80000:90001]
            time_slice = time[80000:90001]
        else:
            x_slice = x[40000:45001]
            F_slice = Fz[40000:45001]
            time_slice = time[40000:45001]
        """
        period_th = 2 * np.pi / fib_w_forced

        n_max = 0
        n_min = 0
        n_F_max = 0
        n_F_min = 0
        maxima = []
        minima = []
        maxima_F = []
        minima_F = []
        print("length x =", len(x))

        for i in range(1, len(x_slice)-1):

            x_previous = x_slice[i-1]
            x_current = x_slice[i]
            x_next = x_slice[i+1]
            if local_maxima(x_previous, x_current, x_next):
                n_max += 1
                maxima.append([x_current, time_slice[i]])
            elif local_minima(x_previous, x_current, x_next):
                n_min += 1
                minima.append([x_current, time_slice[i]])

            F_previous = F_slice[i-1]
            F_current = F_slice[i]
            F_next = F_slice[i+1]
            if local_maxima(F_previous, F_current, F_next):
                n_F_max += 1
                maxima_F.append([F_current, time_slice[i]])
            elif local_minima(F_previous, F_current, F_next):
                n_F_min += 1
                minima_F.append([F_current, time_slice[i]])

        periods = []
        for i in range(1, n_min):
            t = minima[i][1] - minima[i-1][1]
            periods.append(t)
        period_sim = sum(periods) / len(periods)
        w_forced_sim = 2 * np.pi / period_sim
        error_w_forced = (w_forced_sim - fib_w_forced) * 100 / fib_w_forced
        print("error_w_forced =", error_w_forced, "\n")

        amp_sim = (abs(maxima[0][0]) + abs(minima[0][0])) / 2
        print("amp =", amp)
        print("amp_sim =", amp_sim)
        ph_sh_sim = (maxima_F[0][1] - maxima[0][1]) * w_forced_sim
        if ph_sh_sim > 0:
            ph_sh_sim -= 2*np.pi
        print("ph_sh =", ph_sh)
        print("ph_sh_sim =", ph_sh_sim)
        error_ph_sh = (ph_sh_sim - ph_sh) * 100 / ph_sh

    error_amp = (amp_sim - amp) * 100 / amp

    print("\n------------------ Errors ------------------\n")
    print("fib_n = ", fib_n)
    print("error_amp =", error_amp)
    if r != 0.0:
        print("error_ph_sh =", error_ph_sh)
    print("\n")

    fig, ax = plt.subplots()
    #plt.title('Spider position')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')
    #plt.plot(time, x, color='b')
    #plt.plot(time, x0, color='k')
    #plt.plot(time, x1, color='g')
    if fib_eta == 500.0e3:
        plt.plot(time_slice, x_slice, color='b')
        #plt.plot(time[80000:90001], Fz[80000:90001], color='r')
    else:
        plt.plot(time[40000:45001], x[40000:45001], color='b')
        plt.plot(time[40000:45001], Fz[40000:45001], color='r')
    #plt.plot(time, Fz, color='r')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.rcParams['lines.linewidth'] = 0.5
    string = "./png_files/spider_position_dyn_" + str(int(fib_eta/1000)) + "_" + str(fib_w_forced_ratio) + ".png"
    fig.savefig(string)
    plt.close(fig)


def visualize_bend():
    x = []
    time = []
    stringfile = "./csv_files/visualization_bend_" + str(fib_n) + ".csv"
    with open(stringfile, newline='') as csvfile:
        visualreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in visualreader:
            x.append(abs(float(row[0])))
            time.append(float(row[1]))

    k = 48 * fib_E * fib_I / (fib_L**3)
    m = (48 / np.pi**4) * fib_rho * fib_L * fib_A
    q = g * fib_rho * fib_L * fib_A
    F = abs(fib_BC_Fz)
    c_damp = 48 * fib_eta * fib_I / (fib_L**3)
    x_mean = F / k
    w_0 = np.sqrt(k / m)
    print("w_0 =", w_0)
    zeta = c_damp / (2. * m)
    w_d = np.sqrt(w_0**2 - zeta**2)
    x_mean_l = np.array(len(x) * [x_mean])
    x_plus = np.array([x_mean*(1.+np.exp(-zeta*t)) for t in time])
    x_minus = np.array([x_mean*(1.-np.exp(-zeta*t)) for t in time])

    def local_maxima(x1, x2, x3):
        if x1 < x2 and x3 < x2:
            return True
        else:
            return False

    def local_minima(x1, x2, x3):
        if x1 > x2 and x3 > x2:
            return True
        else:
            return False

    n_max = 0
    n_min = 0
    maxima = []
    minima = []
    print("length x =", len(x))
    for i in range(10, len(x)-1):
        x_previous = x[i-1]
        x_current = x[i]
        x_next = x[i+1]
        if local_maxima(x_previous, x_current, x_next):
            n_max += 1
            maxima.append([x_current, time[i]])
        elif local_minima(x_previous, x_current, x_next):
            n_min += 1
            minima.append([x_current, time[i]])

    error_plus = 0
    error_minus = 0
    for i in range(n_max):
        error_plus += ((maxima[i][0] - x_mean*(1.+np.exp(-zeta*maxima[i][1]))) / x_mean)**2
    for i in range(n_min):
        error_minus += ((minima[i][0] - x_mean*(1.-np.exp(-zeta*minima[i][1]))) / x_mean)**2

    error_rms_x_plus = np.sqrt((1. / n_max) * error_plus)
    error_rms_x_minus = np.sqrt((1. / n_min) * error_minus)

    periods = []
    for i in range(1, n_min):
        t = minima[i][1] - minima[i-1][1]
        periods.append(t)
    period_sim = sum(periods) / len(periods)
    w_d_sim = 2 * np.pi / period_sim
    error_w_d = (w_d_sim - w_d) / w_d

    print("###################### ERRORS ######################\n")
    print("n_max = ", n_max)
    print("n_min = ", n_min)
    print("w_d = ", w_d, "w_d_sim = ", w_d_sim)
    print("n = ", fib_n,": error_rms_x_plus = ", error_rms_x_plus)
    print("n = ", fib_n,": error_rms_x_minus = ", error_rms_x_minus)
    print("n = ", fib_n,": error_w_d = ", error_w_d)
    print("Underdamped harmonic oscillator (c < 2*sqrt(k·m)): ", c_damp, " < ", 2*np.sqrt(k*m))

    fig, ax = plt.subplots()
    #plt.title('Middle point position')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')
    plt.plot(time, x, color='b')
    plt.plot(time, x_plus, color='g', linestyle='dashed')
    plt.plot(time, x_minus, color='r', linestyle='dashed')
    plt.plot(time, x_mean_l, color='k', linestyle='dashed')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.rcParams['lines.linewidth'] = 0.1
    string = "./png_files/middle_point_position_bend_" + str(fib_n) + ".png"
    fig.savefig(string)


if __name__ == "__main__":

    ############################################################################
    lfib_n = 27*[8]
    lfib_eta = 9*[500.0e3] + 9*[8500.0e3] + 9*[76500.0e3]
    #lfib_n = 18*[4]
    #lfib_eta = 9*[8500.0e3] + 9*[76500.0e3]
    lfib_w_forced_ratio = [0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0,
                            0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0,
                            0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    #lfib_w_forced_ratio = [ #0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0,
    #                        0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0,
    #                        0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0 ]

    fib_BC_Fz = -500.
    ############################################################################

    fib_rho = 7800.
    fib_d = 6e-3
    fib_L = 0.2
    fib_E = 210.0e9
    g = -9.80665
    fib_m0 = 1
    fib_n_threads = 1
    nu = 0.005428
    dt = 0.000001
    T = 0.05
    L = [2.*np.pi, 2.*np.pi, 2.*np.pi]

    ############################################################################
    #for i in range(len(lfib_n)):
    i = 4
    fib_n = lfib_n[i]
    fib_eta = lfib_eta[i]
    fib_w_forced_ratio = lfib_w_forced_ratio[i]

    fib_BC_index_F = fib_n // 2
    nu = fib_L / fib_n
    fib_L0 = fib_L / fib_n
    fib_n_plus = fib_n + 1 # number of mass points including th spider
    fib_A = np.pi * fib_d**2 / 4
    fib_I = np.pi * fib_d**4 / 64
    fib_m_other = fib_rho * fib_L * fib_A / fib_n
    fib_ratioLd = fib_L / fib_d
    fib_BC_index_fix = fib_n
    fib_k = fib_E * fib_A / fib_L
    fib_m = fib_m0 + fib_rho * fib_A * fib_L / 3
    fib_w_0 = np.sqrt(fib_k / fib_m)
    fib_w_forced = fib_w_forced_ratio * fib_w_0
    print("###################### RESULTS ", str(int(fib_eta/1000)) + "  " + str(fib_w_forced_ratio), " ######################\n")
    #visualize_linstat()
    visualize_lindyn()
    #visualize_bend()
