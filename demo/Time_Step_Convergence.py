import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fib_n =str(20)
N = str(128)
dts = [str(0.005), str(0.0025), str(0.00125), str(0.000625), str(0.0003125),
        str(0.00015625), str(0.001)]

for dt in dts:

    stringfile1 = "./Results/fiber_x1_fn" + fib_n + "_N" + N + "_dt" + dt + ".csv"
    stringfile2 = "./Results/fiber_x2_fn" + fib_n + "_N" + N + "_dt" + dt + ".csv"
    stringfile3 = "./Results/fiber_x3_fn" + fib_n + "_N" + N + "_dt" + dt + ".csv"

    file1 = np.genfromtxt(stringfile1, delimiter=',')
    file2 = np.genfromtxt(stringfile2, delimiter=',')
    file3 = np.genfromtxt(stringfile3, delimiter=',')

    time = file1[..., 0]
    x1_0 = file1[..., 1]
    x2_0 = file2[..., 1]
    x3_0 = file3[..., 1]

    print("\nNew figure for " + dt +":\n", np.shape(time))
    print(np.shape(x1_0))
    print(np.shape(x2_0))
    print(np.shape(x3_0))

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(time, x1_0, color='blue')
    plt.plot(time, x2_0, color='green')
    plt.plot(time, x3_0, color='red')
    plt.savefig("./Convergence/Time_Step_Convergence_dt" + dt + ".png")
    plt.close(fig)

#------------------------------------------------------------------------------#
