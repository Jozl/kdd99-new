import random

from sklearn.neighbors import KNeighborsClassifier

from Code.Dataset.data import Data, DataType, transform_num_to_str
from Code.Dataset.dataset import MyDataSet
from Code.Utils.classify_helper import get_X_y


class Smote:
    def __init__(self, data_name, target_class, under_sampling=None):
        dataset = MyDataSet(data_name, target_class=target_class)
        self.datalist_encoded = [d for d in dataset.datalist]
        self.datalist = [dataset.decode(d) for d in self.datalist_encoded]
        X, y = get_X_y(self.datalist)
        distance, self.itneighbors = KNeighborsClassifier().fit(X, y).kneighbors(self.datalist)
        self.dataset = dataset
        self.attrtype_dict = dataset.attrtype_dict
        self.attrlists = [d.attrlist for d in self.datalist]
        self.classlists = [d.dataclass for d in self.datalist]
        self.data_class = target_class
        self.under_sampling = under_sampling

    def next(self):
        rand_data = random.randint(0, self.dataset.datalist.__len__() - 1)
        rand_attr = random.randint(0, self.itneighbors[rand_data].__len__() - 1)

        # print('before')
        # print(self.dataset.datalist[rand_data].attrlist)

        attrlist_new = [(attr + random.randint(0, 1) * (attr_rand - attr))
                        for attr, attr_rand in
                        zip(self.datalist_encoded[rand_data],
                            self.datalist_encoded[
                                self.itneighbors[rand_data][rand_attr]])]  # 大一些
        # print('after')
        # print(attrlist_new)
        # print(self.dataset.decode(attrlist_new))

        data_new = Data(self.dataset.decode(attrlist_new), dataclass=self.data_class)
        # data_new = Data(attrlist_new, dataclass=self.data_class)

        if self.under_sampling == 'enn':
            clf = KNeighborsClassifier(3)
            clf.fit(self.attrlists, self.classlists)

            if self.data_class != clf.predict([data_new, ]):
                data_new = self.next()

        if self.under_sampling == 'rsb':
            similarityValue = 0.4
            similarityThreshold = 0.9

            for data_true in self.attrlists:
                similarityMatrix = sum([
                    1 - abs((data_true[i] - data_new[i]) / (self.dataset.data_max[i] - self.dataset.data_min[i]))
                    for i in range(data_new.__len__())
                    if not (self.dataset.data_max[i] == self.dataset.data_min[i])]) / data_new.__len__()

            while similarityMatrix >= similarityValue:
                if similarityValue >= similarityThreshold:
                    data_new = self.next()
                    break
                similarityValue += 0.05

        return data_new
