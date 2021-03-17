from deepforest import CascadeForestClassifier


def deep_forest(X_train, X_test, ytrain, ytest):

    df = CascadeForestClassifier(random_state=0, n_estimators=4, n_trees=500, n_jobs=-1, n_tolerant_rounds=3)
    df.fit(X_train, ytrain)
    y_pred = df.predict(X_test)

    return y_pred
