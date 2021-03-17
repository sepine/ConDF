import json
import datasets
from pre_data import split_train_test_label, z_score, split_dataset
from warnings import filterwarnings
from measures import *
from methods import *
from feature_selection import FSelector, get_feature_idx
from pandas import DataFrame

filterwarnings('ignore')

FEATURES_NUM = 89
CLUSTERS_NUM = 14

STEPS = 50

measures_name = ['Accuracy', 'Recall', 'Precision', 'F1', 'F_negative',
                 'G_measure', 'G_mean', 'Bal', 'MCC', 'AUC']

ensemble_names = [
                    # ensemble methods
                    'balanced_random_forest', 'easy_ensemble',
                    'bagging', 'balanced_bagging', 'adaBoost', 'rusBoost',
                    ]

data_columns = ['ST01',	'ST02',	'ST03',	'ST04',	'ST05',	'ST06',	'ST07',	'ST08',	'ST09',	'ST10',	'ST11',
                'CT01',	'CT02',	'CT03',	'CT04',	'CT05',	'CT06',	'CT07',	'CT08',	'CT09',	'CT10',	'CT11',	'CT12',	'CT13',	'CT14',	'CT15',	'CT16',	'CT17',	'CT18',	'CT19',	'CT20',	'CT21',	'CT22',	'CT23',
                'AT01',	'AT02',	'AT03',	'AT04',	'AT05',	'AT06',	'AT07',	'AT08',	'AT09',	'AT10',	'AT11',	'AT12',	'AT13',	'AT14',	'AT15',	'AT16',
                'CB01',	'CB02',	'CB03',	'CB04',	'CB05',	'CB06',	'CB07',	'CB08',	'CB09',	'CB10',	'CB11',	'CB12',	'CB13',	'CB14',	'CB15',	'CB16',	'CB17',	'CB18',	'CB19',	'CB20',	'CB21',	'CB22',	'CB23',
                'AB01',	'AB02',	'AB03',	'AB04',	'AB05',	'AB06',	'AB07',	'AB08',	'AB09',	'AB10',	'AB11',	'AB12',	'AB13',	'AB14',	'AB15',	'AB16',
                'flag']

feature_names = ['chiSquare', 'infoGain', 'gainRatio',
                 'reliefF', 'oneR', 'cfsSubSet', 'consistencySubset']


def build_algo_name(feature_names, ensemble_names):
    algo_names = []
    for feature in feature_names:
        for ensemble in ensemble_names:
            algo = feature + '_' + ensemble
            algo_names.append(algo)
    return algo_names


algo_names = build_algo_name(feature_names, ensemble_names)

# 加载数据集
for da in datasets.d.keys():  # For each dataset
    print(da)
    df = datasets.d[da]

    feature_values = {da: {algo: {i: {measure: []
                                       for measure in measures_name}
                                   for i in range(STEPS)}
                            for algo in algo_names}}

    count = 0

    while count != STEPS:

        try:
            print("数据集" + da + "第" + str(count) + "次循环")

            # 划分数据集
            train, test = split_dataset(df)
            Xtrain, Xtest, ytrain, ytest = split_train_test_label(train, test)

            # 标准化
            x_train_scaled, x_test_scaled = z_score(Xtrain, Xtest)

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

                for ensemble in ensemble_names:
                    y_test_pred = eval(ensemble)(final_train_feature, ytrain, final_test_feature)
                    algo = feature + '_' + ensemble
                    for measure in measures_name:
                        feature_values[da][algo][count][measure] = \
                            str(eval(measure)(ytest, y_test_pred))

        except BaseException as err:
            print('第' + str(count) + '次循环出错')
            print(err)
            count -= 1
        finally:
            count += 1

        continue

    with open('./outputs/ensemble/ensemble--final--' + str(STEPS) + '--' + da + '.json', 'w') as f:
        json.dump(feature_values, f)

