from sklearn.utils import resample

def bagging(X_train, y_train, T=500):
    classifiers = []
    for t in range(T):
        # Bootstrap sample
        X_sample, y_sample = resample(X_train, y_train, n_samples=len(X_train))
        # Train a fully grown decision tree
        clf = ID3(X_sample, X_sample, list(X_sample.columns), target_attribute_name='y', max_depth=None)
        classifiers.append(clf)
    return classifiers

# Use this function to calculate the training and test error for bagging
def calculate_bagging_error(X, y, classifiers):
    final_predictions = np.zeros(len(y))
    for clf in classifiers:
        predictions = np.array([predict(query, clf) for query in X.to_dict(orient="records")])
        final_predictions += predictions
    
    final_predictions = np.round(final_predictions / len(classifiers))
    return np.mean(final_predictions != y)

def bias_variance_decomposition(X_train, y_train, X_test, y_test, T=500, n_repeats=100):
    bagged_trees = []
    single_trees = []
    
    for _ in range(n_repeats):
        # Bootstrap sample
        X_sample, y_sample = resample(X_train, y_train, n_samples=1000)
        # Train bagged trees
        bagged_clf = bagging(X_sample, y_sample, T)
        bagged_trees.append(bagged_clf)
        # Get first fully expanded tree as single tree
        single_clf = ID3(X_sample, X_sample, list(X_sample.columns), target_attribute_name='y', max_depth=None)
        single_trees.append(single_clf)
    
    # Calculate bias and variance for single tree learners
    bias_single, variance_single = calculate_bias_variance(single_trees, X_test, y_test)
    # Calculate bias and variance for bagged trees
    bias_bagging, variance_bagging = calculate_bias_variance(bagged_trees, X_test, y_test)

    return bias_single, variance_single, bias_bagging, variance_bagging
