import numpy as np
import matplotlib.pyplot as plt

def local_regression(x0, X, Y, tau):
    x0 = [1, x0]   
    X = [[1, i] for i in X]
    X = np.asarray(X)
    xw = (X.T) * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau**2)))
    beta = (np.linalg.pinv(xw @ X)) @ (xw @ Y)   
    return (beta @ x0)   

def draw(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.plot(X, Y, 'o', color='black')
    plt.plot(domain, prediction, color='red')
    plt.show()

X = np.linspace(-3, 3, num=1000)#evenly spaced numbers over [-3,3] totally num(1000) numbers
domain = X
Y = np.log(np.abs((X ** 2) - 1) + .5)#just creating y values...chosing this to get W shaped curve

draw(10)
