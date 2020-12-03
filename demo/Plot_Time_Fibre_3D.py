import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import numpy as np

fib_n =str(20)
N = str(128)
dt = str(0.001)

stringfile1 = "./Results/fiber_x1_fn" + fib_n + "_N" + N + "_dt" + dt + ".csv"
stringfile2 = "./Results/fiber_x2_fn" + fib_n + "_N" + N + "_dt" + dt + ".csv"
stringfile3 = "./Results/fiber_x3_fn" + fib_n + "_N" + N + "_dt" + dt + ".csv"

file1 = np.genfromtxt(stringfile1, delimiter=',')
file2 = np.genfromtxt(stringfile2, delimiter=',')
file3 = np.genfromtxt(stringfile3, delimiter=',')

fig = plt.figure()
ax = plt.axes(projection="3d")
xdata, ydata = [], []
ln, = ax.plot([], [], color='blue')
n = 0

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_zlim(0, 2*np.pi)
    return ln,

def update(frame):
    xdata = file1[n, 1:]
    ydata = file2[n, 1:]
    zdata = file3[n, 1:]
    ln.set_data(xdata, ydata, zdata)
    n += 1
    return ln,

ani = FuncAnimation(fig, update, frames=time, init_func=init, blit=True)
plt.show()

#plt.savefig("./Convergence/Time_Step_Convergence_dt" + dt + ".png")
#plt.close(fig)
