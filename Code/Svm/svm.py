import math

import copy
import sys

import numpy as np
from imblearn.combine import SMOTEENN

from sklearn import svm
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

from Code.Dataset.data import DataType
from Code.Dataset.dataset import MyDataSet
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
    learning_rate = 0.000921

    dataset = MyDataSet(data_name, target_class=target_class, encode=True)
    trainer = Trainer(module_features=(dataset.single_continuous_data_len, 30, 20, 16), learning_rate=learning_rate,
                      batch_size=64,
                      dataset=dataset, output_data_label=target_class, output_data_size=target_num)
    trainer(80)
    return trainer.output_data


def gen_with_multi_vae(target_class, target_num, data_name):
    dataset = MyDataSet(data_name, target_class=target_class, encode=True)
    module_features = (dataset.single_continuous_data_len, 30, 20, 16)
    lr = 0.00088
    batch_size = 100

    trainer = Trainer(module_features=module_features, learning_rate=lr,
                      batch_size=batch_size,
                      dataset=dataset, output_data_label=target_class, output_data_size=target_num // 2)(100)

    temp = trainer.output_data
    print(temp[0].attr_list)
    dataset = MyDataSet(temp, target_class=target_class, encode=True)
    trainer = Trainer(module_features=module_features, learning_rate=lr,
                      batch_size=batch_size,
                      dataset=dataset, output_data_label=target_class, output_data_size=target_num)(100)

    return [data.to_list(DataType.CONTINUOUS) for data in trainer.output_data]


def compute_TP_TN_FP_TN(class_test, class_predict, positive, negative):
    TP, TN, FP, FN = 0, 0, 0, 0
    if isinstance(negative, str):
        for x, y in zip(class_test, class_predict):
            TP += x == y == positive
            TN += x == y == negative
            FP += x == negative != y
            FN += x == positive != y
    if isinstance(negative, list):
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


def binary_classify(data_train_total, positive, negative, pos_len, neg_len, expend, data_map, data_name,
                    using_kdd99=False, vae_only=False):
    if expend:
        print('expending-------------------------------------------------------')
        neg_len = pos_len - neg_len
        # vae_predict = gen_with_multi_vae(negative, neg_len, data_name)
        vae_predict = gen_with_vae(negative, neg_len, data_name)

        if using_kdd99:
            smote = MySmote(data_name, target_class=negative, data_map=data_map)
            smote_predict = smote.predict(
                target_len=neg_len, data_map=data_map)

            for l in range(smote_predict.__len__()):
                for i in range(len(smote_predict[l].attr_list)):
                    if smote_predict[l][i] == 'None':
                        smote_predict[l][i] = vae_predict[l][i]
        else:
            smote_predict = vae_predict

        data_predict = []
        for p in smote_predict:
            res = fill_with_eucli_distance(data_train_total, p, data_map)
            data_predict.append(res)
        data_train_total.extend(data_predict)

    y = [d.data_class for d in data_train_total]
    if using_kdd99:
        positive = get_kdd99_big_classification(positive)
        negative = get_kdd99_big_classification(negative)
        y = [get_kdd99_big_classification(c) for c in y]

    if vae_only:
        data_train_total = [d.discrete_to_num(data_map=data_map).to_list([DataType.CONTINUOUS]) for d in
                            data_train_total]
    data_train_total = [d.discrete_to_num(data_map=data_map).attr_list for d in data_train_total]
    # Todo:
    x = np.array(data_train_total).astype(np.float)

    print({k: y.count(k) for k in y})

    # Todo: kf
    kf = KFold(n_splits=5, shuffle=True)
    args = []

    for i_train, i_test in kf.split(X=x, y=y):
        train_x = [x[i] for i in i_train]
        train_y = [y[i] for i in i_train]
        test_x = [x[i] for i in i_test]
        test_y = [y[i] for i in i_test]
        clf = svm.SVC(kernel='linear', probability=True,
                      random_state=np.random.RandomState(0))
        # clf = svm.SVC()
        # clf = GaussianNB()
        # clf = tree.DecisionTreeClassifier()
        clf.fit(train_x, train_y)

        predict_y = [i for i in clf.predict(test_x)]

        temp = compute_classification_indicators(*compute_TP_TN_FP_TN(test_y, predict_y, positive, negative))
        if args.__len__() == 0:
            args = temp
        else:
            args = [a + t for a, t in zip(args, temp)]
    # print(
    #     u'acc\u208A: {:>2.3f}, acc\u208B: {:>2.3f}, accuracy: {:>2.3f}, precision: {:>2.3f}, recall: {:>2.3f}, F1: {:>2.3f}, G-mean: {:>2.3f}'
    #         .format(*[a / 5 for a in args]))
    print(
        u'acc+: {:>2.3f}, acc-: {:>2.3f}, accuracy: {:>2.3f}, precision: {:>2.3f}, recall: {:>2.3f}, F1: {:>2.3f}, G-mean: {:>2.3f}'
            .format(*[a / 5 for a in args]))
    print('')


def multi_classify(data_train_total, positive, negatives, pos_len, negs_len, data_map, data_name, expend=False,
                   using_kdd99=False):
    if expend:
        print('-' * 16 + 'expending' + '-' * 16)
        for negative, neg_len in zip(negatives, negs_len):
            neg_len = pos_len - neg_len
            vae_predict = gen_with_vae(negative, neg_len, data_name)

            if using_kdd99:
                smote = MySmote(data_name, target_class=negative, data_map=data_map)
                smote_predict = smote.predict(
                    target_len=neg_len, data_map=data_map)

                for l in range(smote_predict.__len__()):
                    for i in range(len(smote_predict[l].attr_list)):
                        if smote_predict[l][i] == 'None':
                            smote_predict[l][i] = vae_predict[l][i]
            else:
                smote_predict = vae_predict

            data_predict = []
            for p in smote_predict:
                res = fill_with_eucli_distance(data_train_total, p, data_map)
                data_predict.append(res)
            data_train_total.extend(data_predict)

    y = [d.data_class for d in data_train_total]
    if using_kdd99:
        y = [get_kdd99_big_classification(c) for c in y]
        # TODO PRINT
        print({big_class: y.count(big_class) for big_class in set(y)})

    data_train_total = [d.discrete_to_num(data_map=data_map).attr_list for d in data_train_total]

    x = np.array(data_train_total).astype(np.float)

    kf = KFold(n_splits=5, shuffle=True)

    ones = copy.deepcopy(negatives) + [positive]
    if using_kdd99:
        ones = list(set([get_kdd99_big_classification(c) for c in ones]))
    ones.sort()

    for one in ones:
        negs = [o for o in ones if o != one]
        print('{} vs others'.format(one))

        TP, TN, FP, FN = 0, 0, 0, 0
        acc = 0
        precision = 0
        for i_train, i_test in kf.split(X=x, y=y):
            train_x = [x[i] for i in i_train]
            train_y = [y[i] for i in i_train]
            test_x = [x[i] for i in i_test]
            test_y = [y[i] for i in i_test]
            # clf = svm.SVC()
            # if expend:
            clf = svm.SVC(kernel='linear')
            clf.fit(train_x, train_y)

            predict_y = [i for i in clf.predict(test_x)]

            TP_k, TN_k, FP_k, FN_k = compute_TP_TN_FP_TN(class_test=test_y, class_predict=predict_y, positive=one,
                                                         negative=negs)
            TP += TP_k
            TN += TN_k
            FP += FP_k
            FN += FN_k

        acc = compute_classification_indicators(TP, TN, FP, FN)[0]
        # TODO PRINT
        # print('{:>2.3f}'.format(TP / (TP + TN + FP + FN)))
        print('acc : {:>2.3f}'.format(acc / 5))


def do_classify(data_name='new_data.dat', binary_class=True, using_kdd99=False):
    # TODO print
    print('------------------------------------')

    dataset = MyDataSet(data_name)
    class_dict = dataset.class_dict

    # TODO print
    print(data_name)
    print(class_dict)

    pos_len = 0
    positive = ''
    for k, v in class_dict.items():
        if v > pos_len:
            pos_len = v
            positive = k

    class_dict.pop(positive)
    negatives = list(class_dict.keys())
    negs_len = list(class_dict.values())

    data_map = [{k: 0} for k in dataset.data_list_discrete[0]]
    for data in dataset.data_list_discrete:
        new_data = []
        for attr, attr_map in zip(data, data_map):
            if attr not in attr_map:
                attr_map[attr] = list(attr_map.values())[-1] + 1
            new_data.append(attr_map[attr])

    # 二分类
    if binary_class:
        for negative, neg_len in zip(negatives, negs_len):
            print('{} vs {}'.format(positive, negative))
            data_train_total = copy.deepcopy(dataset.data_list_total)
            binary_classify(data_train_total=copy.deepcopy(dataset.data_list_total), positive=positive,
                            negative=negative, pos_len=pos_len, neg_len=neg_len, expend=False, using_kdd99=using_kdd99,
                            data_name=data_name,
                            data_map=data_map, vae_only=False)
            binary_classify(data_train_total=copy.deepcopy(dataset.data_list_total), positive=positive,
                            negative=negative, pos_len=pos_len, neg_len=neg_len, expend=True, using_kdd99=using_kdd99,
                            data_name=data_name,
                            data_map=data_map, vae_only=False)
    # 多分类
    if not binary_class:
        multi_classify(data_train_total=copy.deepcopy(dataset.data_list_total), positive=positive,
                       negatives=negatives, pos_len=pos_len, negs_len=negs_len, expend=False, using_kdd99=using_kdd99,
                       data_name=data_name, data_map=data_map)
        multi_classify(data_train_total=copy.deepcopy(dataset.data_list_total), positive=positive,
                       negatives=negatives, pos_len=pos_len, negs_len=negs_len, expend=True, using_kdd99=using_kdd99,
                       data_name=data_name, data_map=data_map)


if __name__ == '__main__':
    do_classify(data_name='glass5.dat', binary_class=True, using_kdd99=False)
