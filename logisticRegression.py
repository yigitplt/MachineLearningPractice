from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import copy, math

def load_data():
    data = load_breast_cancer()
    x = data.data
    y = data.target

   

    
    y = y.reshape(-1, 1)

    
    m = x.shape[0]
    train_size = int(0.8 * m)  
    test_size = m - train_size

    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[train_size:]
    y_test = y[train_size:]

    
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_test, y_test

def sigmoid(z):
    return 1/(1+np.exp(-z))


def predict(x, w, b):
    z = np.dot(x, w) + b
    return sigmoid(z)

def compute_cost(x, y, w, b, lambda_):
    m = x.shape[0]
    z = np.dot(x, w) + b
    a = sigmoid(z)
    cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / m
    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)
    return cost + reg_cost


def compute_gradient(x, y, w, b, lambda_):
    m, n = x.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.

    z = np.dot(x, w) + b
    a = sigmoid(z)

    dj_dw = np.dot(x.T, (a - y)) / m  
    dj_db = np.sum(a - y) / m

    dj_dw += (lambda_ / m) * w

    return dj_dw, dj_db

def gradient_descent(X, y, w_initial, b_initial,lambda_, alpha, num_iters):

    J_hist = []
    w = copy.deepcopy(w_initial)
    b = b_initial

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y ,w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b, lambda_)

        if(i < 10000):
            J_hist.append(cost)

        if (i % 1000) == 0:
            print(f"Iteration: {i}, Cost: {cost}")

    return w, b, J_hist

def plot_cost(J_hist):
    plt.plot(J_hist)
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()
   
    w_initial = np.zeros((x_train.shape[1], 1))  
    b_initial = 0.0
    alpha = 0.15
    lambda_ = 0.1
    iterations = 30000

    
    w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_initial, b_initial, lambda_, alpha, iterations)

    
    predictions = predict(x_test, w_final, b_final)
    predicted_labels = (predictions >= 0.5).astype(int)  

    accuracy = np.mean(predicted_labels == y_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    m_test, _ = x_test.shape
    for i in range(20):
        print(f"test prediction: {predicted_labels[i]}, target value: {y_test[i]}")

    print(f"Test Accuracy: {accuracy:.2f}%")

    
    plot_cost(J_hist)

             





