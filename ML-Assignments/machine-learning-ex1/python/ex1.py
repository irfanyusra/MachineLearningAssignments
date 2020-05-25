import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ========================= functions ===================


def plotData(x, y):
    print("here")
    plt.plot(x, y, 'rx', markersize=5, label="training set")
    # m, b = np.polyfit(x, y, 1)
    # plt.plot(x, m*x + b, 'go', label="xlinear regression")
    plt.legend()
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    return plt


def computeCost(x, y, theta):

    m = y.shape[0]
    Hyp = x.dot(theta)  # multiplying matrices
    diff = y - Hyp
    # print(diff)
    squared = np.square(diff)
    # print(squared)
    sumedUp = np.sum(squared)
    J = sumedUp / (2 * m)
    return J


def gradientDescent(x, y, theta, alpha, iterations):
    m = y.shape[0]
    J_history = np.zeros(iterations)

    for iter in range(0, iterations):

        hyp = x.dot(theta)

        Sum = np.transpose(x).dot(hyp - y)
        theta = theta - ((alpha/m) * Sum)
        # print(theta.shape)
        j = computeCost(x, y, theta)

        J_history[iter] = computeCost(x, y, theta)

    print(J_history)
    return (theta, J_history)


# # ==================== Part 1: Basic Function ====================
# print('Running warmUpExercise ... \n');
# print('5x5 Identity Matrix: \n');
# a=np.eye(5)
# print(a)
#  ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv('ex1data1.txt', sep=",", header=None)

data.columns = ["x", "y"]
array = np.array(data)
x = data["x"]
y = data["y"]
m = array.shape[0]  # number of training examples
# print(m)
xarr = np.array(x).reshape(m, 1)
# y = y[:, np.newaxis]   -- same thing below
yarr = np.array(y).reshape(m, 1)


# X = data.iloc[:, 0]  # read first column
# print(X)
# y = data.iloc[:, 1]  # read second column
# print(y)
# m = len(y)  # number of training example
# data.head()  # view first few rows of the data

# Plot Data
plotData(x, y)

# X = X[:, np.newaxis]
# X = np.array(X)
# print(np.newaxis)
# print(X)

# =================== Part 3: Cost and Gradient descent ===================
ones = np.ones((m,1))

# ar = np.append(ones,xarr,1)
xarr = np.hstack((ones, xarr))

# given
theta = np.zeros((2, 1))
iterations = 2000
alpha = 0.01

print('Testing the cost function ...\n')
J = computeCost(xarr, yarr, theta)
print('With theta = [0 ; 0]\nCost computed = ', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
theta = np.array([-1, 2]).reshape(2, 1)
J = computeCost(xarr, yarr, theta)
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J)
print('Expected cost value (approx) 54.24\n')

theta = np.zeros((2,1))
print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, J = gradientDescent(xarr, yarr, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n', theta)

print('Expected theta values (approx): ')
print(' -3.6303\n  1.1664\n\n')
# returns the second column of xarr which we joined with zeros
# print(xarr[:, 1])
# or i could just use the x we had before
plt.plot(x, xarr.dot(theta), label="linear regression")

plt.legend()
plt.show()

# print(theta.shape)
# Predict values for population sizes of 35, 000 and 70, 000
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of \n',
      predict1 * 10000)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of \n',
      predict2 * 10000)

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
# print('Visualizing J(theta_0, theta_1) ...\n')
# fig = plt.figure()
# ax = plt.axes(projection='3d')


# # Data for a three-dimensional line
# theta0_vals = np.linspace(-10, -10, 100)
# theta1_vals = np.linspace(-1, 4, 100)

# J_vals = np.zeros((theta0_vals.shape[0])* (theta1_vals.shape[0]))

# J_vals = J_vals.reshape((theta0_vals.shape[0]),(theta1_vals.shape[0]))
# # print (J_vals)

# # print(theta0_vals[0])

# # Data for three-dimensional scattered points

# # ax.plot3D(theta0_vals, theta1_vals, J_vals, 'gray')
# for i in range (1, (theta0_vals.shape[0])):
    
#     for j in range(1, (theta1_vals.shape[0])):
#         t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(2, 1)
#         J_vals[i][j] = computeCost(xarr, yarr, t)

# J_vals = np.transpose(J_vals);

# print(theta0_vals)
# print(theta1_vals)
# print(J_vals)

# ax.plot3D(np.logspace(-2, 3, 20), np.logspace(-2, 3, 20),
#           np.logspace(-2, 3, 20), 'gray')
# # ax.scatter3D(theta0_vals, theta1_vals, J_vals, c=J_vals, cmap='Greens')
# ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# # ax.plot_wireframe(theta0_vals, theta1_vals, J_vals, color='black')



# ax.set_title('wireframe')
# plt.show()
# # ax.scatter3D(theta0_vals, theta1_vals, J_vals, c=J_vals, cmap='Greens')
