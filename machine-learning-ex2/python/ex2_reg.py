import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# ================= functions=====================


def mapFeature(X1, X2):
    degree = 6
    # print(X1.shape)
    out = np.ones(X1.shape[0])[:, np.newaxis]  # same as reshaping
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),
                                              np.power(X2, j))[:, np.newaxis]))
    return out


def sigmoid(x):
    return 1/(1+np.exp(-x))


def costFunctionReg(theta, X, y, lmbda):
    m = len(y)
    # J = (-1/m) * (y_t.T @ np.log(sigmoid(X_t @ theta_t)) +
    #               (1 - y_t.T) @ np.log(1 - sigmoid(X_t @ theta_t)))

    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))
    J = np.sum(temp1 + temp2) / (-m)
    reg = (lmbda/(2*m)) * (theta[1:].T @ theta[1:])
    J = J + reg
    return J


def gradientDescentReg(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros([m, 1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    return grad


def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y.flatten())
    return(acc * 100)


def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack(
                (out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out


# ================= main =========================
# ----------- Getting data -----------------
data = pd.read_csv('ex2data2.txt', sep=",", header=None)
X = data.iloc[:, 0:2]  # read first two columns into X
y = data.iloc[:, 2]  # read the third column into y
m = len(y)  # no. of training samples
data.head()


# ---------- Plotting data ------------------
# mask = y == 1
# passed = plt.scatter(X[mask][0].values, X[mask][1].values)
# failed = plt.scatter(X[~mask][0].values, X[~mask][1].values)
# plt.xlabel('Microchip Test1')
# plt.ylabel('Microchip Test2')
# plt.legend((passed, failed), ('Passed', 'Failed'))
# # plt.show()


# sending first and second feature columns
X = mapFeature(X.iloc[:, 0], X.iloc[:, 1])

(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n, 1))
lmbda = 1
cost = costFunctionReg(theta, X, y, lmbda)
grad = gradientDescentReg(theta, X, y, lmbda)
print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693\n')
print(
    'Gradient at initial theta (zeros) - first five values only: \n', grad[0:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

# Optimizing theta
output = opt.fmin_tnc(
    func=costFunctionReg,
    x0=theta.flatten(),
    fprime=gradientDescentReg,
    args=(X, y.flatten(), lmbda))
theta_opt = output[0]
print(theta_opt)  # theta contains the optimized values


acc = accuracy(X, y.flatten(), theta_opt, 0.5)

print('Train Accuracy: ', acc)
print('Expected accuracy (with lambda = 1) (approx): 83.1')

# Predict probability --------------------------------
tests = np.array([[0.45],[0.45]])

test2 = np.array([tests[1]])
tests = mapFeature(tests[0], tests[1])
prob = sigmoid(tests.dot(theta_opt))
print(
    'using the optimized theta from fmin using the initial theta of [0, 0, 0] ')
print('we predict a yes probability of ', prob)


u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))


for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta_opt)
mask = y.flatten() == 1
X = data.iloc[:, :-1]
passed = plt.scatter(X[mask][0], X[mask][1])
failed = plt.scatter(X[~mask][0], X[~mask][1])
plt.contour(u, v, z, 0)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()


