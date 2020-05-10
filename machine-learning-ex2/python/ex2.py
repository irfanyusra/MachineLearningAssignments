import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize as opt    # for min functions


# ========================= functions ===================

# ------ sigmoid used in classification problems to get predictions between 0 and 1
def sigmoid(z):
    return 1/(1+np.exp(-z))


def costFunction(theta, X, y):
    # x = np.array([1, 2, 3])
    # ya = np.array([2, 3, 4])
    # print(x @ ya)
    # print(x.dot(ya))
    # print(np.matmul(x, ya))
    # print(x*ya)
    # print(np.multiply(x, ya))

    # J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta)))
    #     + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))

    # @ is basically np.matmultiply or .dot in this case? --- i think

    J = (1/m)*(np.sum((-y * (np.log(sigmoid(X.dot(theta))))) -
                      ((1-y) * (np.log(1 - sigmoid(X.dot(theta)))))))

    return (J)


def gradient(theta, X, y):
    return (1 / m) * ((X.T).dot((sigmoid(X.dot(theta)) - y)))
    # return ((1/m) * X.T @ (sigmoid(X @ theta) - y))


def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    return(acc * 100)

# =============================== main ===================


# Load Data
data = pd.read_csv('ex2data1.txt', sep=",", header=None)

X = data.iloc[:, 0:2]  # read first two columns into X
y = data.iloc[:, 2]  # read the third column into y
m = len(y)  # no. of training samples
data.head()

# plot shows diff colors using mask
mask = y == 1
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
# plt.show()

# ============ Part 2: Compute Cost and Gradient ============

# costFunction (0,X,y)

(m, n) = X.shape
X = np.hstack((np.ones((m, 1)), X))
y = y[:, np.newaxis]


theta = np.zeros((n+1, 1))  # intializing theta with all zeros
cost = costFunction(theta, X, y)
grad = gradient(theta, X, y)
print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):\n ', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# intializing theta with all zeros
theta = (np.array([-24, 0.2, 0.2])).reshape(3, 1)
cost = costFunction(theta, X, y)
grad = gradient(theta, X, y)
print('Cost at test theta: ', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:\n ', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')


# ============= Part 3: Optimizing using fminunc  =============
# In this exercise, you will use a built-in function (fminunc) to find the optimal parameters theta.

initial_theta = np.zeros((n+1, 1))  # intializing theta with all zeros
# initial_theta = (np.array([-24, 0.2, 0.2])).reshape(3, 1)

# print (y.flatten())
temp = opt.fmin_tnc(func=costFunction,
                    x0=initial_theta.flatten(), fprime=gradient,
                    args=(X, y.flatten()))
# the output of above function is a tuple whose first element contains the optimized values of theta
theta_optimized = temp[0]
# print(theta_optimized)

J = costFunction(theta_optimized[:, np.newaxis], X, y)

# Print theta to screen
print('Cost at theta found by fminunc: ', J)
print('Expected cost (approx): 0.203')
print('theta: \n', theta_optimized)
print('Expected theta (approx):\n -25.161\n 0.206\n 0.201\n')


plot_x = [np.min(X[:, 1]-2), np.max(X[:, 2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0]
                                + np.dot(theta_optimized[1], plot_x))
mask = y.flatten() == 1
fig = plt.figure()

adm = plt.scatter(X[mask][:, 1], X[mask][:, 2])
not_adm = plt.scatter(X[~mask][:, 1], X[~mask][:, 2])
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))

# plt.show()


# ============== Part 4: Predict and Accuracies ==============

acc = accuracy(X, y.flatten(), theta_optimized, 0.5)

print('Train Accuracy: ', acc)
print('Expected accuracy (approx): 89.0')


# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
marks = np.array([1, 45, 85])
prob = sigmoid(marks.dot(theta_optimized))
print(
    'using the optimized theta from fmin using the initial theta of [0, 0, 0] ')
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
print('Expected value: 0.775 +/- 0.002\n')
