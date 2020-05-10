import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize as opt    # for min functions
from scipy.io import loadmat
import functions as fn #all functions for this ex



# =========== Part 1: Loading and Visualizing Data =============
# Load Data

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

_, axarr = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axarr[i, j].imshow(X[np.random.randint(X.shape[0])].
                           reshape((20, 20), order='F'))
        axarr[i, j].axis('off')

# plt.show()

# Initialization

m = len(y)
ones = np.ones((m, 1))
X = np.hstack((ones, X))  # add the intercept
(m, n) = X.shape
num_labels = 10


# ============ Part 2a: Vectorize Logistic Regression ============
# fprintf('\nTesting lrCostFunction() with regularization');

# theta_t = [-2; -1; 1; 2];
# X_t = [ones(5,1) reshape(1:15,5,3)/10];
# y_t = ([1;0;1;0;1] >= 0.5);
# lambda_t = 3;
# [J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

# fprintf('\nCost: %f\n', J);
# fprintf('Expected cost: 2.534819\n');
# fprintf('Gradients:\n');
# fprintf(' %f \n', grad);
# fprintf('Expected gradients:\n');
# fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

# ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lmbda = 0.1
theta = fn.thetaOneVsAll(X, y, num_labels, lmbda,n)

# ----------- Part 3: Predict for One-Vs-All -----------

acc = fn.predictOneVsAll(theta, X,y)
print('Training Set Accuracy: ', acc)
