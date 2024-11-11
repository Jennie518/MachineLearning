import random

def random_forest(X_train, y_train, T=500, feature_subset_size=2):
    classifiers = []
    for t in range(T):
        # Bootstrap sample
        X_sample, y_sample = resample(X_train, y_train, n_samples=len(X_train))
        # Randomly select a subset of features
        feature_subset = random.sample(list(X_sample.columns), feature_subset_size)
        # Train a decision tree using the feature subset
        clf = ID3(X_sample, X_sample, feature_subset, target_attribute_name='y', max_depth=None)
        classifiers.append(clf)
    return classifiers


def bias_variance_random_forest(X_train, y_train, X_test, y_test, T=500, feature_subset_size=2, n_repeats=100):
    random_forests = []
    
    for _ in range(n_repeats):
        # Bootstrap sample
        X_sample, y_sample = resample(X_train, y_train, n_samples=1000)
        # Train random forest
        forest_clf = random_forest(X_sample, y_sample, T, feature_subset_size)
        random_forests.append(forest_clf)

    # Calculate bias and variance for random forests
    bias_rf, variance_rf = calculate_bias_variance(random_forests, X_test, y_test)
    
    return bias_rf, variance_rf
