def get_X_y(datalist):
    return [d.attrlist for d in datalist], [d.dataclass for d in datalist]


def compute_TP_TN_FP_FN(class_test, class_predict, positive, negative):
    TP, TN, FP, FN = 0, 0, 0, 0
    for x, y in zip(class_test, class_predict):
        TP += x == y == positive
        TN += x == y == negative
        FP += x == negative != y
        FN += x == positive != y

    return TP, TN, FP, FN


def compute_classification_indicators(TP, TN, FP, FN):
    acc_p, acc_n, accuracy, precision, recall, F1, G_mean = 0, 0, 0, 0, 0, 0, 0
    try:
        acc_p = TP / (TP + FN)
    except ZeroDivisionError:
        pass
    try:
        acc_n = TN / (TN + FP)
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

    return acc_p, acc_n, accuracy, precision, recall, F1, G_mean
