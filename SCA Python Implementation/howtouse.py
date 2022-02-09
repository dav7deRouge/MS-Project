import scipy.io as spio
import numpy as np
import SCA 

# Extracting data
mat = spio.loadmat('data.mat')

# How to use example
target = 4
vals = [2, 3]
source = [0, 1]
data_cell = mat['XY_cell']
# Target data:
X_t = data_cell[0][target][:, 0:2]
Y_t = data_cell[0][target][:, 2]
X_t = np.asarray(X_t); Y_t = np.asarray(Y_t, dtype=int)
# Source data:
X_s = []; Y_s = []
for i in source:
  X_s.append(data_cell[0][i][:, 0:2])
  Y_s.append(data_cell[0][i][:, 2])
X_s = np.asarray(X_s); Y_s = np.asarray(Y_s, dtype=int)
# Validation data:
X_v = []; Y_v = []
for i in vals:
  X_v.append(data_cell[0][i][:, 0:2])
  Y_v.append(data_cell[0][i][:, 2])
X_v = np.asarray(X_v); Y_v = np.asarray(Y_v, dtype=int)
X_v = np.reshape(X_v, (X_v.shape[0]*X_v.shape[1], X_v.shape[2]))
Y_v = np.reshape(Y_v, (Y_v.shape[0]*Y_v.shape[1]))

params = { 'beta': [0.1, 0.3, 0.5, 0.7, 0.9],
            'delta': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
            'k_list': [2],
            'X_v': X_v,
            'Y_v': Y_v }
test_accuracy, predicted_labels, Z_s, Z_t = SCA.sca(X_s, Y_s, X_t, Y_t, params)