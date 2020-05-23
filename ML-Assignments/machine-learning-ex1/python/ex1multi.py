# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ========================= functions ===================


def featureNormalize(X):
    mu = np.mean(X)
    sigma = np.std(X)
    X = (X - mu)/sigma
    return (X, mu, sigma)


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    #same as the other one 
    computeCostMulti(X, y, theta)
    J_history = np.zeros((num_iters, 1))
    m = len(y)
    # print(y)
    for iter in range(0, num_iters):
        hyp = X.dot(theta)
        Sum = np.transpose(X).dot(hyp - y)
        theta = theta - ((alpha/m) * Sum)
        J_history[iter] = computeCostMulti(X, y, theta)

    return (theta, J_history)


def computeCostMulti(X, y, theta):
    #same as the other one 
    m = len(y)
    J = 0
    Hyp = X.dot(theta)  # multiplying matrices
    diff = y - Hyp
    # print(diff)
    squared = np.square(diff)
    # print(squared)
    sumedUp = np.sum(squared)
    J = sumedUp / (2 * m)

    return J


def normalEqn(X, y):
    theta = np.zeros((X.shape[1], 1))
    xt = np.transpose(X)
    temp = np.linalg.inv(xt.dot(X))
    theta = (temp.dot(xt)).dot(y)
    # theta = *X'*y
    return theta



#  == == == == == == == == Part 1: Feature Normalization == == == == == == == ==
# Load Data
data = pd.read_csv('ex1data2.txt', sep=",", header=None)

# ---------One way of extracting data-------
# data.columns = ["x1", "x2","y"]
# array = np.array(data)
# x1 = data["x1"]
# x2 = data["x2"]
# y = data["y"]
# m = array.shape[0]  # number of training examples
# print(m)
# x1arr = np.array(x1).reshape(m, 1)
# x2arr = np.array(x2).reshape(m, 1)
# xarr = np.hstack((x1arr, x2arr))
# print (xarr)

# -------------second way ----------------

X = data.iloc[:, 0:2]  # read first two columns into X
y = data.iloc[:, 2]  # read the third column into y
m = len(y)  # no. of training samples
data.head()


# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = featureNormalize(X)

ones = np.ones((m, 1))
X = np.hstack((ones, X))

y = y[:, np.newaxis]  # into np array shape (m,1)
alpha = 0.01
num_iters = 400
theta = np.zeros((3, 1))

# same as the single variable one 
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
print('Theta computed from gradient descent: ', theta);

xval = (np.arange(0,len(J_history))).reshape(len(J_history),1)
plt.plot(xval, J_history, label="linear regression")
plt.xlabel('Number of iterations');
plt.ylabel('Cost J');
plt.legend()


J = computeCostMulti(X, y, theta)
print(J)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ', J);


# ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n');

data = pd.read_csv('ex1data2.txt', sep=",", header=None)

X = data.iloc[:, 0:2]  # read first two columns into X
y = data.iloc[:, 2]  # read the third column into y
m = len(y)  # no. of training samples
data.head()
ones = np.ones((m, 1))
X = np.hstack((ones, X))

theta = normalEqn(X, y);
print('Theta computed from the normal equations: ', theta);

prediction = np.array([1,1650, 3]);
price = prediction.dot(theta);
print(prediction)
print(price)






# plt.show()




