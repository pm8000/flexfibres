import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('NS_isotropic_128_128_128_w.h5', 'r')
print('Name of the file:\n')
print(f.name)
print('Keys stored in the file:\n')
print(list(f.keys()))
group_P = f['P']  # P group
subgroup_3D_P = group_P['3D']  # 3D subgroup
dataset_P = subgroup_3D_P['15000'] # '15000' dataset
#print(dataset_P.shape)
#print(dataset_P.dtype)

group_U0 = f['U0']  # U0 group
subgroup_3D_U0 = group_U0['3D']  # 3D subgroup
dataset_U0 = subgroup_3D_U0['15000'] # '15000' dataset
#print(dataset_U0.shape)
#print(dataset_U0.dtype)

group_U1 = f['U1']  # U1 group
subgroup_3D_U1 = group_U1['3D']  # 3D subgroup
dataset_U1 = subgroup_3D_U1['15000'] # '15000' dataset
#print(dataset_U1.shape)
#print(dataset_U1.dtype)

group_U2 = f['U2']  # U2 group
subgroup_3D_U2 = group_U2['3D']  # 3D subgroup
dataset_U2 = subgroup_3D_U0['15000'] # '15000' dataset
#print(dataset_U2.shape)
#print(dataset_U2.dtype)

list = []
for i in range(0,127):
    for j in range(0,127):
        list.append([i, j, dataset_P[i,j,64], dataset_U0[i,j,64], dataset_U1[i,j,64], dataset_U2[i,j,64]])
velocity_2D_slice = np.array(list)
np.savetxt('test_paraview.csv', velocity_2D_slice, fmt='%4.10f', delimiter=',')
# Headers:  x, y, P, U0, U1, U2
