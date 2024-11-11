def stochastic_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6):
    n, m = X.shape
    w = np.zeros(m)
    cost_history = []
    
    for i in range(1000):  # Set a max iteration limit
        for j in range(n):
            random_index = np.random.randint(n)
            X_i = X[random_index, :].reshape(1, m)
            y_i = y[random_index].reshape(1)
            predictions = X_i.dot(w)
            error = predictions - y_i
            gradient = X_i.T.dot(error)
            w_new = w - learning_rate * gradient.flatten()
            
            cost = np.sum((X.dot(w) - y) ** 2) / (2 * n)
            cost_history.append(cost)
            
            if np.linalg.norm(w_new - w) < tolerance:
                return w, cost_history
            w = w_new
    
    return w, cost_history

def analytical_solution(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
