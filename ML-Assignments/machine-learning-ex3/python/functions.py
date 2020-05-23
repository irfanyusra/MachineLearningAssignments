import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize as opt    # for min functions
from scipy.io import loadmat


def sigmoid(z):
    return 1/(1+np.exp(-z))


def costFunctionReg(theta, X, y, lmbda):
    m = len(y)
    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))
    return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * lmbda / (2*m)


def gradRegularization(theta, X, y, lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * lmbda / m
    temp[0] = temp[0] - theta[0] * lmbda / m
    return temp


def thetaOneVsAll(X, y, num_labels, lmbda, n):

    theta = np.zeros((num_labels, n))  # inital parameters
    for i in range(num_labels):
        if i == 0:
            digit_class = 10
        else:
            digit_class = i
        theta[i] = opt.fmin_cg(
            f=costFunctionReg,
            x0=theta[i],
            fprime=gradRegularization,
            args=(X, (y == digit_class).flatten(), lmbda),
            maxiter=50)
    return theta


def predictOneVsAll(theta, X, y):

    hyp = sigmoid(X.dot(theta.T))
    pred = np.argmax(hyp, axis=1)
    pred = [e if e else 10 for e in pred]
    acc = np.mean(pred == y.flatten()) * 100
    return acc


def predict(Theta1, Theta2, X, y):
    # print ("X",X)
    # print ("Theta1",Theta1)
    # print("Theta2",Theta2)

    # (m, n) = X.shape

    # (num_labels, _) = Theta2.shape
    # pred = np.zeros((m, 1))

    # ones = np.ones((m, 1))
    # temp1 = np.hstack((ones, X))
    # temp2 = np.hstack((ones, (sigmoid(temp1.dot(np.transpose(Theta1))))))
    # # print("here2", (temp1 @(np.transpose(Theta1)))[:, 0])
    # temp3 = sigmoid(temp2.dot(Theta2.T))
    # print (temp3[:,0])
    # pred = np.argmax(temp3, axis=1)
    # acc = np.mean(pred == y.flatten()) * 100

    # return (pred, acc)
