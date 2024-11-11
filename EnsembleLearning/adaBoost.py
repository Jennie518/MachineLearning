# Modify your decision tree learning algorithm to learn decision stumps (two-level trees)
def decision_stump(X, y, weights):
    features = X.columns
    best_feature = None
    max_gain = float('-inf')
    for feature in features:
        gain = info_gain(X, feature, y)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature
    return best_feature

def adaBoost(X_train, y_train, T=500):
    n = X_train.shape[0]
    weights = np.ones(n) / n
    classifiers = []
    alphas = []
    for t in range(T):
        # Train a decision stump
        stump_feature = decision_stump(X_train, y_train, weights)
        stump_tree = ID3(X_train, X_train, [stump_feature], target_attribute_name='y', max_depth=1)

        # Predict on training set
        predictions = np.array([predict(query, stump_tree) for query in X_train.to_dict(orient="records")])
        misclassified = predictions != y_train

        # Calculate error rate
        error = np.dot(weights, misclassified) / sum(weights)

        # Calculate alpha
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
        alphas.append(alpha)

        # Update weights
        weights *= np.exp(-alpha * y_train * (2 * predictions - 1))
        weights /= sum(weights)

        classifiers.append(stump_tree)

    return classifiers, alphas

# Use this to calculate the training and test error for each iteration T
def calculate_adaBoost_error(X, y, classifiers, alphas):
    final_predictions = np.zeros(len(y))
    for i, clf in enumerate(classifiers):
        predictions = np.array([predict(query, clf) for query in X.to_dict(orient="records")])
        final_predictions += alphas[i] * (2 * predictions - 1)

    final_predictions = np.sign(final_predictions)
    return np.mean(final_predictions != y)
