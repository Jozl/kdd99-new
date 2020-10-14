import math

import copy
import os
import sys
import time

from sklearn import svm
from sklearn.model_selection import KFold

from Code.Dataset.data import DataType
from Code.Dataset.dataset import MyDataSet
from Code.Utils.dataset_creator import dataset_create
from Code.Dataset.kdd99 import get_kdd99_big_classification
from Code.Smote.smote import MySmote
from Code.Vae.trainer import Trainer


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
    dataset = MyDataSet(data_name, target_class=target_class, encode=True)

    learning_rate = 0.000921
    module_features = (dataset.single_continuous_data_len, 30, 16)
    batch_size = 128

    trainer = Trainer(module_features=module_features, learning_rate=learning_rate,
                      batch_size=batch_size,
                      dataset=dataset, output_data_label=target_class, output_data_size=target_num)
    trainer(60)
    return trainer.output_data


def gen_with_smote(target_class, target_num, data_name, discrete_num_map):
    smote = MySmote(data_name, target_class=target_class, data_map=discrete_num_map)
    smote_predict = smote.predict(target_len=target_num, data_map=discrete_num_map)

    return smote_predict


def compute_TP_TN_FP_TN(class_test, class_predict, positive, negative):
    TP, TN, FP, FN = 0, 0, 0, 0
    for x, y in zip(class_test, class_predict):
        TP += x == y == positive
        TN += x == y and x in negative
        FP += x in negative and x != y
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


def multi_classify(data_train, positive, negatives, positive_len, negative_lens, discrete_num_map, data_name,
                   expend=False, ):
    if expend:
        print('\033[0;36m--expending' + '-' * 16 + '\033[0m')
        for negative, negative_len in zip(negatives, negative_lens):

            negative_len = math.floor((positive_len - sum([l for n, l in zip(negatives, negative_lens)
                                                           if get_kdd99_big_classification(
                    n) == get_kdd99_big_classification(negative)])) *
                                      (negative_len / sum([l for n, l in zip(negatives, negative_lens)
                                                           if get_kdd99_big_classification(
                                              n) == get_kdd99_big_classification(negative)])))

            vae_predict = gen_with_vae(negative, negative_len, data_name)
            smote_predict = gen_with_smote(negative, negative_len, data_name, discrete_num_map)

            for l in range(smote_predict.__len__()):
                for i in range(len(smote_predict[l].attr_list)):
                    if smote_predict[l][i] == 'None':
                        smote_predict[l][i] = vae_predict[l][i]

            data_predict = []
            for p in smote_predict:
                res = fill_with_eucli_distance(data_train, p, discrete_num_map)
                data_predict.append(res)
            data_train.extend(data_predict)

    X = [d.discrete_to_num(discrete_num_map).attr_list for d in data_train]
    y = [get_kdd99_big_classification(d.data_class) for d in data_train]

    print({big_class: y.count(big_class) for big_class in set(y)})
    kf = KFold(n_splits=5, shuffle=True)

    big_classes = copy.deepcopy(negatives) + [positive]
    big_classes = list(set([get_kdd99_big_classification(c) for c in big_classes]))
    big_classes.sort()

    clf = svm.SVC()
    # clf = GaussianNB()
    # clf = tree.DecisionTreeClassifier()
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
    #                          algorithm="SAMME",
    #                          n_estimators=200, learning_rate=0.8)
    print('\nusing : ', clf.__class__.__name__)
    for target_class in big_classes:
        print('\033[0;33m {} vs rest \033[0m'.format(target_class))
        rest_classes = [bc for bc in big_classes if bc != target_class]

        acc = None
        for i_train, i_test in kf.split(X, y):
            train_X = [X[i] for i in i_train]
            train_y = [y[i] for i in i_train]
            test_X = [X[i] for i in i_test]
            test_y = [y[i] for i in i_test]

            clf.fit(train_X, train_y)

            predict_y = [i for i in clf.predict(test_X)]

            if acc:
                acc += \
                    compute_classification_indicators(
                        *compute_TP_TN_FP_TN(test_y, predict_y, target_class, rest_classes))[
                        0]
            else:
                acc = \
                    compute_classification_indicators(
                        *compute_TP_TN_FP_TN(test_y, predict_y, target_class, rest_classes))[
                        0]
        print('acc : {:>2.3f}'.format(acc / 5))


def do_classify(data_name):
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

    positive_len = max(len_list)
    negative_lens = [l for l in len_list if l != positive_len]
    positive = reversed_class_dict[positive_len]
    negatives = [reversed_class_dict[l] for l in len_list if l != positive_len]

    discrete_num_map = [{k: 0} for k in dataset.data_list_discrete[0]]
    for data in dataset.data_list_discrete:
        for attr, attr_map in zip(data, discrete_num_map):
            if attr not in attr_map:
                attr_map[attr] = list(attr_map.values())[-1] + 1

    # 多分类
    multi_classify(data_train=copy.deepcopy(dataset.data_list_total), positive=positive,
                   negatives=negatives, positive_len=positive_len, negative_lens=negative_lens, expend=False,
                   data_name=data_name, discrete_num_map=discrete_num_map)
    multi_classify(data_train=copy.deepcopy(dataset.data_list_total), positive=positive,
                   negatives=negatives, positive_len=positive_len, negative_lens=negative_lens, expend=True,
                   data_name=data_name, discrete_num_map=discrete_num_map)


def main():
    data_name = dataset_create(
        ['normal', 'neptune', 'ftp_write', 'warezclient', 'rootkit', 'nmap'],
        # (400, 70, 8, 12, 20, 6)
        (1000, 88, 8, 12, 20, 181)
    )
    do_classify(data_name)


if __name__ == '__main__':
    file_name = str(os.path.basename(__file__)).replace('py', 'txt')

    from Code.Utils.stdout_catch import Logger

    sys.stdout = Logger(file_name)

    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
