import math

import copy
import sys

from sklearn import svm
from sklearn.model_selection import KFold

from Code.Dataset.data import DataType
from Code.Dataset.dataset import MyDataSet
from Code.Utils.dataset_creator import dataset_create
from Code.Dataset.kdd99 import get_kdd99_big_classification
from Code.Smote.smote import MySmote
from Code.Vae.trainer import Trainer

'''
    二分类准确度计算
        在 kfold 的时候保证分出的每一块都有真实的负类数据
'''


def fill_with_eucli_distance(original_data_list, predict_data, data_map):
    closest = []
    eucli_distance = sys.maxsize
    pd = predict_data.discrete_to_num(data_map).to_list([DataType.CONTINUOUS, DataType.DISCRETE])

    for original_data in original_data_list:
        od = original_data.discrete_to_num(data_map).to_list([DataType.CONTINUOUS, DataType.DISCRETE])
        dist = math.sqrt(sum([(p - o) ** 2 for p, o in zip(pd, od) if p != 'None']))

        if dist < eucli_distance:
            eucli_distance = dist
            closest = original_data

    for i in range(len(predict_data)):
        if predict_data[i] == 'None':
            predict_data[i] = closest[i]

    return predict_data


def gen_with_vae(target_class, target_num, data_name):
    learning_rate = 0.000921

    dataset = MyDataSet(data_name, target_class=target_class, encode=True)
    trainer = Trainer(module_features=(dataset.single_continuous_data_len, 30, 20, 16), learning_rate=learning_rate,
                      batch_size=64,
                      dataset=dataset, output_data_label=target_class, output_data_size=target_num)
    trainer(80)

    return trainer.output_data


def gen_with_smote(target_class, target_num, data_name, discrete_num_map):
    smote = MySmote(data_name, target_class=target_class, data_map=discrete_num_map)
    smote_predict = smote.predict(target_len=target_num, data_map=discrete_num_map)

    return smote_predict


def compute_TP_TN_FP_TN(class_test, class_predict, positive, negative):
    TP, TN, FP, FN = 0, 0, 0, 0
    for x, y in zip(class_test, class_predict):
        TP += x == y == positive
        TN += x == y == negative
        FP += x == negative != y
        FN += x == positive != y

    # TODO PRINT
    # print('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP, TN, FP, FN))

    return TP, TN, FP, FN


def compute_classification_indicators(TP, TN, FP, FN):
    acc_p, acc_m, accuracy, precision, recall, F1, G_mean = 0, 0, 0, 0, 0, 0, 0
    try:
        acc_p = TP / (TP + FN)
    except ZeroDivisionError:
        pass
    try:
        acc_m = TN / (TN + FP)
    except ZeroDivisionError:
        pass
    try:
        accuracy = (TP + TN) / (TP + FN + FP + TN)
    except ZeroDivisionError:
        pass
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        pass
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        pass
    try:
        F1 = (2 * TP) / (2 * TP + FN + FP)
    except ZeroDivisionError:
        pass
    try:
        G_mean = ((TP / (TP + FN)) * (TN / (TN + FP))) ** 0.5
    except ZeroDivisionError:
        pass

    return acc_p, acc_m, accuracy, precision, recall, F1, G_mean


def binary_classify(data_train, positive, negative, positive_len, negative_len, discrete_num_map, data_name, ):
    print('\033[0;36m--expending' + '-' * 16 + '\033[0m')
    negative_len = positive_len - negative_len

    vae_predict = gen_with_vae(negative, negative_len, data_name)
    smote_predict = gen_with_smote(negative, negative_len, data_name, discrete_num_map)

    for l in range(smote_predict.__len__()):
        for i in range(len(smote_predict[l].attr_list)):
            if smote_predict[l][i] == 'None':
                smote_predict[l][i] = vae_predict[l][i]

    data_predict = [fill_with_eucli_distance(data_train, predict, discrete_num_map)
                    for predict in smote_predict]

    data_sp = [d for d in data_train if d.data_class == negative]
    for d in data_sp:
        data_train.remove(d)

    X_sp = [d.discrete_to_num(discrete_num_map).attr_list for d in data_sp]
    y_sp = [get_kdd99_big_classification(d.data_class) for d in data_sp]

    data_train.extend(data_predict)
    X = [d.discrete_to_num(discrete_num_map).attr_list for d in data_train]
    y = [get_kdd99_big_classification(d.data_class) for d in data_train]

    # TODO PRINT
    print({k: y.count(k) for k in y})

    positive = get_kdd99_big_classification(positive)
    negative = get_kdd99_big_classification(negative)

    # Todo: kf
    kf = KFold(n_splits=5, shuffle=True)
    args = None
    for (i_train, i_test), (i_sp_train, i_sp_test) in zip(kf.split(X, y), kf.split(X_sp, y_sp)):
        train_X = [X[i] for i in i_train] + [X_sp[i] for i in i_sp_train]
        train_y = [y[i] for i in i_train] + [y_sp[i] for i in i_sp_train]
        test_X = [X[i] for i in i_test] + [X_sp[i] for i in i_sp_test]
        test_y = [y[i] for i in i_test] + [y_sp[i] for i in i_sp_test]
        # clf = svm.SVC(kernel='linear', probability=True,
        #               random_state=np.random.RandomState(0))
        clf = svm.SVC()
        # clf = GaussianNB()
        # clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_y)

        predict_y = [i for i in clf.predict(test_X)]

        if args:
            args = [a + t for a, t in zip(args, compute_classification_indicators(
                *compute_TP_TN_FP_TN(test_y, predict_y, positive, negative)))]
        else:
            args = compute_classification_indicators(*compute_TP_TN_FP_TN(test_y, predict_y, positive, negative))
    print(
        'acc+: {:>2.3f}, acc-: {:>2.3f}, accuracy: {:>2.3f}, precision: {:>2.3f}, recall: {:>2.3f}, F1: {:>2.3f}, '
        'G-mean: {:>2.3f} '
            .format(*[arg / 5 for arg in args]))


def do_classify(data_name='kdd99_binary_test.dat'):
    # TODO print
    print('------------------------------------')

    dataset = MyDataSet(data_name)
    class_dict = dataset.class_dict

    # TODO print
    print(data_name)
    print(class_dict)

    reversed_class_dict = dict(zip(class_dict.values(), class_dict.keys()))
    len_list = list(reversed_class_dict.keys())
    len_list.sort()

    positive_len, negative_len = len_list[1], len_list[0]
    positive_class, negative_class = reversed_class_dict[positive_len], reversed_class_dict[negative_len]

    discrete_num_map = [{k: 0} for k in dataset.data_list_discrete[0]]
    for data in dataset.data_list_discrete:
        for attr, attr_map in zip(data, discrete_num_map):
            if attr not in attr_map:
                attr_map[attr] = list(attr_map.values())[-1] + 1

    # 二分类
    print('{} vs {}'.format(positive_class, negative_class))
    binary_classify(data_train=copy.deepcopy(dataset.data_list_total), positive=positive_class,
                    negative=negative_class, positive_len=positive_len, negative_len=negative_len,
                    data_name=data_name,
                    discrete_num_map=discrete_num_map)


def main(classes=('normal', 'teardrop'), class_lens=(1000, 32)):
    data_name = dataset_create(classes, class_lens)
    do_classify(data_name)


def __call__():
    main()


if __name__ == '__main__':
    main()

