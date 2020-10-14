# -*- coding:utf-8 -*-

# print(__doc__)

import numpy as np
# from scipy import interp
# import matplotlib.pyplot as plt
import os
import math
from sklearn import svm, datasets
from sklearn import datasets, neighbors
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import random
import time

# start = time.clock()
###############################################################################
mark = ['b', 'g', 'r', '--r', '--c', 'm', 'y', 'k', '--b', '--g']
# 加载数据
os.chdir('F:\\sample static\\zximbanlanced\\test1')
files = os.listdir('F:\\sample static\\zximbanlanced\\test1')
# print files
x = 0
for m in range(len(files)):
    start = time.clock()
    print(files[m])

    iris = np.loadtxt(files[m])
    X = np.zeros((len(iris), len(iris[0]) - 1))  # 二维矩阵 50行，4列
    y = np.zeros(len(iris))  # 一维矩阵 50列
    for k1 in range(len(iris)):
        for k2 in range(len(iris[0]) - 1):
            X[k1][k2] = iris[k1][k2]
            y[k1] = iris[k1][-1]

    # n_samples, n_features = X.shape
    # # print X.shape

    random_state = np.random.RandomState(0)  # 随机数生成器
    ###############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)  # 均分
    all_tpr = []

    cv = StratifiedKFold(y, n_folds=10)
    classifier_SVM = svm.SVC(kernel='linear', probability=True,
                             random_state=random_state, C=1)
    # print cv
    for i, (train, test) in enumerate(cv):
        print('================================================')
        # 计算不平衡因子imbalancedFactor
        y_num0 = 0
        y_num1 = 0
        for m1 in range(len(y[train])):
            if y[train][m1] == 1:
                y_num1 = y_num1 + 1
            if y[train][m1] == 0:
                y_num0 = y_num0 + 1
        imbalancedFactor = float(y_num1) / y_num0
        # print imbalancedFactor

        # 将标签为0的训练数据存在矩阵X0中
        X0 = np.zeros((y_num0, len(iris[0]) - 1))
        y0 = np.zeros(y_num0)
        k5 = 0
        for k3 in range(len(y[train])):
            if y[train][k3] == 0:
                for k4 in range(len(iris[0]) - 1):
                    X0[k5][k4] = X[train][k3][k4]
                k5 = k5 + 1
                if k5 == y_num0:
                    break

        # 将标签为1的训练数据存在矩阵X1中
        X1 = np.zeros((y_num1, len(iris[0]) - 1))
        # y1 = np.zeros(y_num1)
        k7 = 0
        for k6 in range(len(y[train])):
            if y[train][k6] == 1:
                for k8 in range(len(iris[0]) - 1):
                    X1[k7][k8] = X[train][k6][k8]
                k7 = k7 + 1
                if k7 == y_num1:
                    break

        # 在X0矩阵中，为每个行样本选择其k个邻居
        k = math.floor(imbalancedFactor)  # kNN中k的取值
        classifier_KNN = neighbors.KNeighborsClassifier(n_neighbors=int(k))
        clf_KNN = classifier_KNN.fit(X0, y0)
        # clf_KNN = classifier_KNN.fit(X[train], y[train])
        distance, itneighbors = clf_KNN.kneighbors(X0, n_neighbors=int(k) + 1)
        # print distance
        # print len(itneighbors)
        # print itneighbors

        # SMOTE扩充少类（0类）矩阵
        X0_new = np.zeros(len(X0[0]))
        for m2 in range(len(X0)):
            new = np.zeros(len(X0[0]))
            for m3 in range(len(itneighbors[0]) - 1):
                for m4 in range(len(X0[0])):
                    # print itneighbors[m2][m3+1]
                    new[m4] = X0[m2][m4] + random.randint(0, 1) * (X0[itneighbors[m2][m3 + 1]][m4] - X0[m2][m4])  # 大一些
                    # X0_new[m4] = X0[m2][m4] + random.random() * (X0[itneighbors[m2][m3+1]][m4] - X0[m2][m4])  #小一些
                # print X0_new
                X0_new = np.vstack((X0_new, new))  # 扩充之后的矩阵，标签为0

        X0 = np.vstack((X0, X0_new))
        # print len(X1)
        # print len(X0)
        y1 = np.ones(len(X1))  # 1标签
        y00 = np.zeros(len(X0))  # 0标签

        clf_SVM = classifier_SVM.fit(np.vstack((X1, X0)), np.hstack((y1, y00)))

        probas_ = clf_SVM.predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        print(roc_auc)

        probas_t = clf_SVM.predict(X[test])
        # print probas_t
        num1 = 0
        num0 = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for k in range(len(probas_t)):
            if probas_t[k] == 1.0:
                num1 = num1 + 1
            if probas_t[k] == 0:
                num0 = num0 + 1
            if probas_t[k] == 1.0 and y[test][k] == 1.0:  # 正-》正
                TP = TP + 1
            if probas_t[k] == 0 and y[test][k] == 1:  # 正-》负
                FN = FN + 1
            if probas_t[k] == 1 and y[test][k] == 0:  # 负-》正
                FP = FP + 1
            if probas_t[k] == 0 and y[test][k] == 0:  # 负-》负
                TN = TN + 1

        # print 'precision ---------------'
        precision = float(TP) / (TP + FP)
        print(precision)

        # print 'recall/accT ---------------'
        recall = float(TP) / (TP + FN)
        print(recall)
        accT = recall

        # print 'accF ---------------'
        accF = float(TN) / (TN + FP)
        print(accF)

        # print 'F1 ---------------'
        F_measure = 2 * float(TP) / (2 * TP + FP + FN)
        print(F_measure)

        # print 'G_mean ---------------'
        G_mean = np.sqrt(float(TP) / (TP + FN) * float(TN) / (FP + TN))
        print(G_mean)

        score = clf_SVM.score(X[test], y[test])
        print(score)
