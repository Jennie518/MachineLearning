def batch_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6):
    n, m = X.shape
    w = np.zeros(m)
    cost_history = []
    
    while True:
        predictions = X.dot(w)
        errors = predictions - y
        gradient = X.T.dot(errors) / n
        w_new = w - learning_rate * gradient
        
        # Calculate cost function
        cost = np.sum(errors ** 2) / (2 * n)
        cost_history.append(cost)
        
        # Check convergence
        if np.linalg.norm(w_new - w) < tolerance:
            break
        
        w = w_new
    
    return w, cost_history
