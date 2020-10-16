import copy

from sklearn.svm import SVC
from sklearn.model_selection import KFold

from Code.Utils.classify_helper import compute_classification_indicators, compute_TP_TN_FP_FN, get_X_y
from Code.Utils.generate_data import *
from Code.Utils.sheet_helper import SheetWriter

c = SheetWriter()


def binary_classify(dataset, positive, negative, positive_len, negative_len, data_name, expend=None, ):
    data_train = copy.deepcopy(dataset.get_original_datalist)
    # 只用连续值
    data_train = [d.to_data(attrtype_dict=dataset.attrtype_dict, attrtype_list=[DataType.CONTINUOUS]) for d in
                  data_train]

    data_original_positive = copy.deepcopy(data_train)

    data_original_negative = [d for d in data_original_positive if d.dataclass == negative]
    # 剔除 真实的negative数据
    data_original_positive = [d for d in data_original_positive if d.dataclass == positive]

    if expend:
        negative_len = positive_len - negative_len
        data_predict = eval('gen_with_{}'.format(expend))(negative, negative_len, data_name, encode=False)
        data_predict = [d.to_data(attrtype_dict=dataset.attrtype_dict, attrtype_list=[DataType.CONTINUOUS]) for d in
                        data_predict]
        data_train.extend(data_predict)
        data_original_positive.extend(data_predict)

        clf_ori = SVC()
        # clf_ori = SVC(kernel='linear')

        X_ori, y_ori = get_X_y(data_original_positive)
        X_ori_test, y_ori_test = get_X_y(data_original_negative)

        acc = 0
        print('训练分类器的数据: ', {k: y_ori.count(k) for k in set(y_ori)})
        kf = KFold(n_splits=5, shuffle=True)
        for (i_train, _), (_, i_test) in zip(kf.split(X_ori, y_ori), kf.split(X_ori_test, y_ori_test)):
            train_X = [X_ori[i] for i in i_train]
            train_y = [y_ori[i] for i in i_train]
            test_X = [X_ori_test[i] for i in i_test]
            test_y = [y_ori_test[i] for i in i_test]

            clf_ori.fit(train_X, train_y)
            predict = clf_ori.predict(test_X)

            TP, TN, FP, FN = compute_TP_TN_FP_FN(test_y, predict, positive, negative)
            acc += TN / (TN + FP)
        # print('\033[0;36moriginal acc :  {:>2.3f} \033[0m'.format(acc))
        print('\033[0;36moriginal acc :  {:>2.3f} \033[0m'.format(acc / 5))

    X, y = get_X_y(data_train)

    print('训练分类器的数据: ', {k: y.count(k) for k in set(y)})
    kf = KFold(n_splits=5, shuffle=True)
    args = []
    for i_train, i_test in kf.split(X, y):
        train_X = [X[i] for i in i_train]
        train_y = [y[i] for i in i_train]
        test_X = [X[i] for i in i_test]
        test_y = [y[i] for i in i_test]

        clf = SVC()
        # clf = SVC(kernel='linear')
        clf.fit(train_X, train_y)

        predict_y = [i for i in clf.predict(test_X)]

        temp = compute_classification_indicators(*compute_TP_TN_FP_FN(test_y, predict_y, positive, negative))
        if args.__len__() == 0:
            args = temp
        else:
            args = [a + t for a, t in zip(args, temp)]
    print(
        u'acc+: {:>2.3f}, acc-: {:>2.3f}, accuracy: {:>2.3f}, precision: {:>2.3f}, recall: {:>2.3f}, F1: {:>2.3f}, G-mean: {:>2.3f}'
            .format(*[a / 5 for a in args]))

    # c.writerow(['acc+', 'acc-', 'accuracy', 'precision', 'recall', 'F1', 'G-mean'])
    output = [round(a / 5, 3) for a in args]
    output.insert(2, '' if not expend else round(acc / 5, 3))
    c.writerow([data_name] + output)


def do_classify(data_name='kdd99_binary_test.dat', expend_alg=None):
    dataset = MyDataSet(data_name)
    dataclass_dict = dataset.dataclass_dict

    print()
    print('\033[0;32mdataset : {} \033[0m'.format(data_name))
    print('\033[0;32mexpend alg : {} \033[0m'.format(expend_alg))
    print(dataclass_dict)

    reversed_class_dict = dict(zip(dataclass_dict.values(), dataclass_dict.keys()))
    len_list = list(reversed_class_dict.keys())
    len_list.sort()

    positive_len, negative_len = len_list[1], len_list[0]
    positive_class, negative_class = reversed_class_dict[positive_len], reversed_class_dict[negative_len]

    print('\ndo {}'.format(expend_alg))
    binary_classify(dataset=dataset, positive=positive_class,
                    negative=negative_class, positive_len=positive_len, negative_len=negative_len,
                    data_name=data_name, expend=expend_alg)


def main(data_name, expend_alg):
    for _ in range(1):
        do_classify(data_name, expend_alg)


if __name__ == '__main__':
    data_names = [
        'yeast-0-5-6-7-9_vs_4.dat',
        'ecoli4.dat',
        'glass5.dat',
        'yeast5.dat',
        'yeast6.dat',
        'kdd99_new_multi.dat',
    ]
    expend_algs = [
        None,
        'vae',
        'smote',
        'smote_enn',
        'smote_rsb',
    ]

    c.writerow(['data_name', 'acc+', 'acc-', 'acc_original', 'accuracy', 'precision', 'recall', 'F1', 'G-mean'])
    for expend_alg in expend_algs:
        c.writerow([''])
        c.writerow(['do nothing' if not expend_alg else expend_alg])
        for data_name in data_names:
            main(data_name, expend_alg)
            pass
