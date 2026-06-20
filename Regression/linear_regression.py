import numpy as np

def fit_linear_regression(X, y, lr=0.05, epoch=8):
    if (X.shape[0] != y.shape[0]):
        raise ValueError("Not correct input dimensions")
    
    n = X.shape[0]
    weights = np.zeros(X.shape[1])
    bias = 0.0
    loss_showcase = []

    for j in range(epoch):
        y_pred = X @ weights + bias
        for i in range(weights.size):
            gradient = (1/n) * np.sum((y_pred - y) * X[:, i])
            weights[i] -= (lr * gradient)
        gradient_b = ((1/n) * np.sum(y_pred - y))
        bias -= (lr * gradient_b)

        loss = (1/n) * np.sum((y_pred - y)**2)
        loss_showcase.append(loss)
        print(f"In the epoch: {j}, the loss is {loss}")

    print(f"The updated weights are {weights} and bias is {bias}")
    return weights, bias, loss_showcase


#More optimal weight and bias gradient calculation is
#dw = (1/n) * (X.T @ (y_pred - y))
#db = (1/n) * np.sum(y_pred - y)
#
#weights -= lr * dw
#bias -= lr * db