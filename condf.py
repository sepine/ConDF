import json
import datasets
from pre_data import split_train_test_label, z_score, split_dataset
from experiment.SCMIC import scmic
from warnings import filterwarnings
from model import deep_forest
from measures import *
from feature_selection import FSelector, get_feature_idx
from experiment.feature import *
from pandas import DataFrame

filterwarnings('ignore')

FEATURES_NUM = 89
CLUSTERS_NUM = 14

STEPS = 50

measures_name = ['Accuracy', 'Recall', 'Precision', 'F1', 'F_negative',
                 'G_measure', 'G_mean', 'Bal', 'MCC', 'AUC']

algorithm_names = [
                    # ensemble methods
                    # 'balanced_random_forest', 'easy_ensemble',
                    # 'bagging', 'balanced_bagging', 'adaBoost', 'rusBoost',
                    'gcforest']

classifier_names = [
                    # classifier after the scmic
                    'scmic_naive_bayes', 'scmic_decision_tree', 'scmic_svm_svc',
                    'scmic_lr', 'scmic_random_forest', 'scmic_nn', 'scmic_nn_3',
                    'scmic_nn_5', 'scmic_nn_7', 'scmic_nn_9',
                    # base classifier
                    # 'naive_bayes', 'decision_tree', 'svm_svc', 'lr', 'random_forest',
                    # 'nn', 'nn_3', 'nn_5', 'nn_7', 'nn_9'
                ]

data_columns = ['ST01',	'ST02',	'ST03',	'ST04',	'ST05',	'ST06',	'ST07',	'ST08',	'ST09',	'ST10',	'ST11',
                'CT01',	'CT02',	'CT03',	'CT04',	'CT05',	'CT06',	'CT07',	'CT08',	'CT09',	'CT10',	'CT11',	'CT12',	'CT13',	'CT14',	'CT15',	'CT16',	'CT17',	'CT18',	'CT19',	'CT20',	'CT21',	'CT22',	'CT23',
                'AT01',	'AT02',	'AT03',	'AT04',	'AT05',	'AT06',	'AT07',	'AT08',	'AT09',	'AT10',	'AT11',	'AT12',	'AT13',	'AT14',	'AT15',	'AT16',
                'CB01',	'CB02',	'CB03',	'CB04',	'CB05',	'CB06',	'CB07',	'CB08',	'CB09',	'CB10',	'CB11',	'CB12',	'CB13',	'CB14',	'CB15',	'CB16',	'CB17',	'CB18',	'CB19',	'CB20',	'CB21',	'CB22',	'CB23',
                'AB01',	'AB02',	'AB03',	'AB04',	'AB05',	'AB06',	'AB07',	'AB08',	'AB09',	'AB10',	'AB11',	'AB12',	'AB13',	'AB14',	'AB15',	'AB16',
                'flag']

feature_names = ['chiSquare', 'infoGain', 'gainRatio',
                 'reliefF', 'oneR', 'cfsSubSet', 'consistencySubset']


# 加载数据集
for da in datasets.d.keys():  # For each dataset
    print(da)
    df = datasets.d[da]

    # 保存指标
    results_json = {da: {i: {measure: []
                               for measure in measures_name}
                           for i in range(STEPS)}}

    algo_values = {da: {algo: {i: {measure: []
                                   for measure in measures_name}
                               for i in range(STEPS)}
                        for algo in algorithm_names}}

    clf_values = {da: {classifier: {i: {measure: []
                                   for measure in measures_name}
                               for i in range(STEPS)}
                        for classifier in classifier_names}}

    feature_values = {da: {feature: {i: {measure: []
                                       for measure in measures_name}
                                   for i in range(STEPS)}
                            for feature in feature_names}}

    count = 0

    while count != STEPS:

        try:
            print("数据集" + da + "第" + str(count) + "次循环")

            # 划分数据集
            train, test = split_dataset(df)
            Xtrain, Xtest, ytrain, ytest = split_train_test_label(train, test)

            # 标准化
            x_train_scaled, x_test_scaled = z_score(Xtrain, Xtest)

            # # 对比算法
            # print('==数据集' + da + '==共循坏' + str(STEPS) + '次==现在是第' + str(count) + '次==对比算法==')
            #
            # for algo_name in algorithm_names:
            #     if algo_name == 'gcforest':
            #         y_test_pred = deep_forest(x_train_scaled, x_test_scaled, ytrain, ytest)
            #     else:
            #         # call the algorithm to predict
            #         y_test_pred = eval(algo_name)(x_train_scaled, ytrain, x_test_scaled)
            #
            #     # calculate the evaluation
            #     for measure in measures_name:
            #         algo_values[da][algo_name][count][measure] = \
            #             str(eval(measure)(ytest, y_test_pred))

            # # 对比分类器
            # print('==数据集' + da + '==共循坏' + str(STEPS) + '次==现在是第' + str(count) + '次==对比分类器==')
            #
            # for classifier in classifier_names:
            #     # call the algorithm to predict
            #
            #     y_test_pred = eval(classifier)(x_train_scaled, ytrain, x_test_scaled)
            #
            #     # calculate the evaluation
            #     for measure in measures_name:
            #         clf_values[da][classifier][count][measure] = \
            #             str(eval(measure)(ytest, y_test_pred))

            # 需要将标签拼接在标准化后的特征的最后一列
            x_train_scaled_label = np.insert(x_train_scaled, FEATURES_NUM, values=ytrain.T, axis=1)
            x_test_scaled_label = np.insert(x_test_scaled, FEATURES_NUM, values=ytest.T, axis=1)

            df_train = DataFrame(x_train_scaled_label, columns=data_columns)
            df_test = DataFrame(x_test_scaled_label, columns=data_columns)

            # 单独的特征选择方法
            print('==数据集' + da + '==共循坏' + str(STEPS) + '次==现在是第' + str(count) + '次==Feature==')

            fs = FSelector(df_train)
            feature_all_idx = fs.all_idx

            for feature in feature_names:

                my_idx = get_feature_idx(feature_names.index(feature), feature_all_idx)

                final_train_feature = x_train_scaled_label.T[my_idx].T
                final_test_feature = x_test_scaled_label.T[my_idx].T

                y_test_pred = deep_forest(final_train_feature, final_test_feature, ytrain, ytest)

                for measure in measures_name:
                    feature_values[da][feature][count][measure] = \
                        str(eval(measure)(ytest, y_test_pred))

        except BaseException as err:
            print('第' + str(count) + '次循环出错')
            print(err)
            count -= 1
        finally:
            count += 1

        continue

    # with open('./outputs/algo--final--' + str(STEPS) + '--' + da + '.json', 'w') as f:
    #     json.dump(algo_values, f)
    #
    # with open('./outputs/clf--final--' + str(STEPS) + '--' + da + '.json', 'w') as f:
    #     json.dump(clf_values, f)
    #
    with open('./outputs/feature--single--final--' + str(STEPS) + '--' + da + '.json', 'w') as f:
        json.dump(feature_values, f)

