import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()  
ax = Axes3D(fig)  
ax.view_init(elev=15, azim=-7)  


grad_mat = np.array([[ 1.0, -0.3, -0.0],
                         [-1.0,  1.0, -0.2],
                         [-1.0, -1.0,  1.0]])

print(grad_mat)
O = np.array([0, 0, 0])
A = grad_mat[0, :]
B = grad_mat[1, :]
C = grad_mat[2, :]

d_all = np.array([[-0.6014802395334626 , -0.21118537877736254 , 0.7704688554649243 , ],
[-0.05106881823742133 , 0.05565476588564567 , 0.9971431807107981 , ],
[-0.7602859212697055 , -0.2280857763809116 , 0.6082287370157644 , ],
[0.18749548573246536 , 0.4247159493317133 , 0.8856984843688107 , ],
])

d_FedFV = d_all[0, :]
d_FedMGDA_plus = d_all[1, :]
d_FedSGD = d_all[2, :]
d_FedMGDP = d_all[3, :]


d_randoms = np.random.rand(200000, 3) * 2 - 1.0
d_randoms = d_randoms / np.linalg.norm(d_randoms, axis=1).reshape(-1, 1)
c = -(grad_mat @ d_randoms.T)
d_descents = d_randoms[np.where(np.all(c <= 1e-5, axis=0))[0], :]
print(d_descents.shape)
shallow_prepare = np.zeros((0, 3))
for i in range(d_descents.shape[0]):
    shallow_prepare = np.vstack([shallow_prepare, np.zeros((1, 3))])
    shallow_prepare = np.vstack([shallow_prepare, d_descents[i, :]])
common_descent_directions = shallow_prepare.T



ax.plot([O[0], A[0]], [O[1], A[1]], [O[2], A[2]], '-', c='black')
ax.plot([O[0], B[0]], [O[1], B[1]], [O[2], B[2]], '-', c='black')
ax.plot([O[0], C[0]], [O[1], C[1]], [O[2], C[2]], '-', c='black')

ax.plot([O[0], d_FedSGD[0]], [O[1], d_FedSGD[1]], [O[2], d_FedSGD[2]], '-', c='red', label='FedSGD')
ax.plot([O[0], d_FedFV[0]], [O[1], d_FedFV[1]], [O[2], d_FedFV[2]], '--', c='red', label='FedFV')
ax.plot([O[0], d_FedMGDA_plus[0]], [O[1], d_FedMGDA_plus[1]], [O[2], d_FedMGDA_plus[2]], ':', c='red', label='FedMGDA+')
ax.plot([O[0], d_FedMGDP[0]], [O[1], d_FedMGDP[1]], [O[2], d_FedMGDP[2]], '-', c='green', label='GPFL')

ax.plot(common_descent_directions[0, :], common_descent_directions[1, :], common_descent_directions[2, :], '-', c='gray', alpha=0.5, label='All possible common descent directions')

x1, x2 = -2, 2
y1, y2 = -2, 2
z1, z2 = -2, 2
ax.set_xlim3d((x1, x2))
ax.set_ylim3d((y1, y2))
ax.set_zlim3d((z1, z2))

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.legend()
plt.show()
