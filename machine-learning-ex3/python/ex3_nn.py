import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize as opt    # for min functions
from scipy.io import loadmat
import functions as fn
# ======= Initialization
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10



# =========== Part 1: Loading and Visualizing Data =============
# Load Data
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
(m, n) = X.shape

# _, axarr = plt.subplots(10, 10, figsize=(10, 10))
# for i in range(10):
#     for j in range(10):
#         axarr[i, j].imshow(X[np.random.randint(X.shape[0])].
#                            reshape((20, 20), order='F'))
#         axarr[i, j].axis('off')

# plt.show()

#  ================ Part 2: Loading Pameters ================

print('\nLoading Saved Neural Network Parameters ...')

data2 = loadmat('ex3weights.mat')
# print(data2)
Theta1 = data2['Theta1']
Theta2 = data2['Theta2']

# print(Theta1)

# ================= Part 3: Implement Predict =================
(pred, acc) = fn.predict(Theta1, Theta2, X, y)
print('\nTraining Set Accuracy:\n',acc)
print (pred)




