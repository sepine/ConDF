import argparse
from lib.gcforest.gcforest import GCForest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 0
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config


def gcforest(X_train, X_test, ytrain, ytest):
    """
    gcforest的封装
    :param X_train:
    :param X_test:
    :param ytrain:
    :param ytest:
    :return:
    """
    config = get_toy_config()

    gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train, y_train = X_train[:2000], y_train[:2000]
    # X_train = final_train[:, np.newaxis, :, :]
    # X_test = final_test[:, np.newaxis, :, :]

    X_train = X_train
    X_test = X_test
    ytrain = ytrain.flatten()
    ytest = ytest.flatten()

    # X_train_enc = gc.fit_transform(X_train, ytrain)

    # X_enc is the concatenated predict_proba result of each estimators of the last layer of the GCForest model
    # X_enc.shape =
    #   (n_datas, n_estimators * n_classes): If cascade is provided
    #   (n_datas, n_estimators * n_classes, dimX, dimY): If only finegrained part is provided
    # You can also pass X_test, y_test to fit_transform method, then the accracy on test data will be logged when training.
    X_train_enc, X_test_enc = gc.fit_transform(X_train, ytrain, X_test=X_test, y_test=ytest)
    # WARNING: if you set gc.set_keep_model_in_mem(True), you would have to use
    # gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test) to evaluate your model.

    y_pred = gc.predict(X_test)

    # You can try passing X_enc to another classfier on top of gcForest.e.g. xgboost/RF.
    # X_test_enc = gc.transform(X_test)
    # X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
    # X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
    # X_train_origin = X_train.reshape((X_train.shape[0], -1))
    # X_test_origin = X_test.reshape((X_test.shape[0], -1))
    # X_train_enc = np.hstack((X_train_origin, X_train_enc))
    # X_test_enc = np.hstack((X_test_origin, X_test_enc))
    # print("X_train_enc.shape={}, X_test_enc.shape={}".format(X_train_enc.shape, X_test_enc.shape))
    # clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
    # clf.fit(X_train_enc, y_train)
    # y_pred = clf.predict(X_test_enc)
    # acc = accuracy_score(y_test, y_pred)
    # print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc * 100))

    # # dump
    # with open("test.pkl", "wb") as f:
    #     pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
    # # load
    # with open("test.pkl", "rb") as f:
    #     gc = pickle.load(f)
    # y_pred = gc.predict(X_test)
    # acc = accuracy_score(ytest, y_pred)
    # print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))

    return y_pred