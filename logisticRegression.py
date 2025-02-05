from sklearn.datasets import load_breast_cancer  
import numpy as np  
import matplotlib.pyplot as plt  
import copy, math  

def load_data():
    """
    Loads and preprocesses the Breast Cancer dataset.
    - Splits the dataset into training and test sets (80% train, 20% test).
    - Normalizes the data by subtracting the mean and dividing by the standard deviation.
    
    Returns:
        x_train, y_train, x_test, y_test: Processed training and test data.
    """  
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

    # Feature scaling: Normalize the features using mean and standard deviation
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_test, y_test

def sigmoid(z):
    """
    Computes the sigmoid function: 1 / (1 + exp(-z)).
    
    Args:
        z: The input value (can be a scalar, vector, or matrix).
        
    Returns:
        The sigmoid of the input.
    """
    return 1 / (1 + np.exp(-z))

def predict(x, w, b):
    """
    Predicts the class labels for input data x using logistic regression.
    
    Args:
        x: Input data (features).
        w: Weight vector (parameters).
        b: Bias term.
        
    Returns:
        The predicted probability of class 1.
    """
    z = np.dot(x, w) + b  
    return sigmoid(z)  

def compute_cost(x, y, w, b, lambda_):
    """
    Computes the cost function for logistic regression with L2 regularization.
    
    Args:
        x: Input data (features).
        y: True labels.
        w: Weight vector (parameters).
        b: Bias term.
        lambda_: Regularization parameter.
        
    Returns:
        The computed cost (log loss with regularization).
    """
    m = x.shape[0]  
    z = np.dot(x, w) + b  
    a = sigmoid(z)  
    
   
    cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / m
    
    
    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)
    
    return cost + reg_cost  

def compute_gradient(x, y, w, b, lambda_):
    """
    Computes the gradient of the cost function with respect to w and b.
    
    Args:
        x: Input data (features).
        y: True labels.
        w: Weight vector (parameters).
        b: Bias term.
        lambda_: Regularization parameter.
        
    Returns:
        dj_dw, dj_db: Gradients of the cost function w.r.t. w and b.
    """
    m, n = x.shape  
    dj_dw = np.zeros((n, ))  
    dj_db = 0.  

    
    z = np.dot(x, w) + b
    a = sigmoid(z)  

    
    dj_dw = np.dot(x.T, (a - y)) / m  
    dj_db = np.sum(a - y) / m  

    
    dj_dw += (lambda_ / m) * w

    return dj_dw, dj_db  

def gradient_descent(X, y, w_initial, b_initial, lambda_, alpha, num_iters):
    """
    Performs gradient descent to minimize the cost function and find optimal weights.
    
    Args:
        X: Input data (features).
        y: True labels.
        w_initial: Initial weight vector.
        b_initial: Initial bias term.
        lambda_: Regularization parameter.
        alpha: Learning rate.
        num_iters: Number of iterations for gradient descent.
        
    Returns:
        w_final, b_final: Optimized weights and bias.
        J_hist: History of the cost function values over iterations.
    """
    J_hist = []  
    w = copy.deepcopy(w_initial)  
    b = b_initial 

    
    for i in range(num_iters):
        
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)

        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        
        cost = compute_cost(X, y, w, b, lambda_)

        # Store the cost for plotting
        if i < 10000:
            J_hist.append(cost)

        
        if (i % 1000) == 0:
            print(f"Iteration: {i}, Cost: {cost}")

    return w, b, J_hist  

def plot_cost(J_hist):
    """
    Plots the history of the cost function over iterations.
    
    Args:
        J_hist: History of cost function values.
    """
    plt.plot(J_hist)  
    plt.title("Cost vs. Iteration")  
    plt.xlabel("Iteration")  
    plt.ylabel("Cost")  
    plt.show()  

if __name__ == "__main__":

    # Load and preprocess the data
    x_train, y_train, x_test, y_test = load_data()

    # Initialize parameters
    w_initial = np.zeros((x_train.shape[1], 1))  
    b_initial = 0.0  
    alpha = 0.15  
    lambda_ = 0.1  
    iterations = 30000 

    # Train the logistic regression model using gradient descent
    w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_initial, b_initial, lambda_, alpha, iterations)

    # Make predictions on the test set
    predictions = predict(x_test, w_final, b_final)
    predicted_labels = (predictions >= 0.5).astype(int)  # Convert probabilities to binary labels

    # Display some predictions and their corresponding true values
    m_test, _ = x_test.shape
    for i in range(20):
        print(f"test prediction: {predicted_labels[i]}, target value: {y_test[i]}")

    # Calculate the accuracy of the model
    accuracy = np.mean(predicted_labels == y_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot the cost function history
    plot_cost(J_hist)
