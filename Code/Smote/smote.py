import random
from collections import Counter

from imblearn.combine import SMOTEENN
from sklearn import neighbors

from Code.Dataset.data import Data, DataType
from Code.Dataset.dataset import MyDataSet


class MySmote:
    def __init__(self, data_name, target_class, data_map):
        dataset = MyDataSet(data_name, encode=False, target_class=target_class)
        self.__dataset = dataset
        self.__target_class = target_class
        data_list = dataset.data_list_discrete
        data_num_list = []
        for data in data_list:
            new_data = []
            for attr, attr_map in zip(data, data_map):
                if attr not in attr_map:
                    attr_map[attr] = list(attr_map.values())[-1] + 1
                new_data.append(attr_map[attr])
            data_num_list.append(new_data)
        self.__data_num_list = data_num_list
        self.__data_map = data_map

    def predict(self, target_len, data_map):
        classifier_knn = neighbors.KNeighborsClassifier()
        clf_knn = classifier_knn.fit(self.__data_num_list, self.__dataset.data_class_list)
        distance, itneighbors = clf_knn.kneighbors(self.__data_num_list)

        data_num_list_expend = []

        for i in range(target_len):
            # TODO
            rand = random.randint(0, self.__data_num_list.__len__() - 1)
            data_new = [(attr + random.randint(0, 1) * (attr_rand - attr))
                        for attr, attr_rand in
                        zip(self.__data_num_list[rand],
                            self.__data_num_list[
                                itneighbors[rand][random.randint(0, itneighbors[rand].__len__() - 1)]])]  # 大一些
            data_num_list_expend.append(data_new)

        return [Data(d, attr_dict=self.__dataset.attr_dict, data_class=self.__target_class, data_type=DataType.DISCRETE)
                for d in MySmote.data_list_num_to_symbol(data_num_list_expend, data_map=data_map)]

    @staticmethod
    def data_list_discrete_to_num(data_list, data_map):
        return [[attr_map[attr] for attr, attr_map in zip(data, data_map)] for data in data_list]

    @staticmethod
    def data_list_num_to_symbol(data_num_list, data_map):
        return [[{v: k for k, v in attr_map.items()}[attr]
                 for attr, attr_map in zip(data, data_map)]
                for data in data_num_list]

    @staticmethod
    def data_num_to_symbol(attr_list, data_map):
        return [{v: k for k, v in attr_map.items()}[attr]
                for attr, attr_map in zip(attr_list, data_map)]

    @property
    def data_map(self):
        return self.__data_map

    @property
    def data_num_list(self):
        return self.__data_num_list
