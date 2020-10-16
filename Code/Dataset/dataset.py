import copy

import numpy as np

import torch
from torch.utils.data import Dataset

from Code.Dataset.data import Data, DataType, str_to_num_dicts
from Code.Dataset.dataloader import DataLoader


class MyDataSet(Dataset):
    def __init__(self, dataname, target_class=None):
        datalist = []
        dataclass_dict = {}
        attrtype_dict = {}

        self.dataname = dataname
        with DataLoader.load(dataname) as data_file:
            for row in data_file:
                if row.startswith('@'):
                    row_split = [r.strip() for r in row.split(' ')]
                    if row.startswith('@attribute'):
                        if row_split[-1].endswith(']'):
                            attrtype_dict[row_split[1]] = DataType.CONTINUOUS
                        elif row_split[-1].endswith('1}') or row_split[-1].endswith('{0}'):
                            attrtype_dict[row_split[1]] = DataType.TWO_VALUE
                        else:
                            attrtype_dict[row_split[1]] = DataType.DISCRETE
                    if row.startswith('@outputs'):
                        attrtype_dict.pop(row_split[1])
                else:
                    d = Data(row, dataclass=target_class)
                    if target_class and target_class != d.dataclass:
                        continue
                    datalist.append(d)
                    if d.dataclass in dataclass_dict:
                        dataclass_dict[d.dataclass] += 1
                    else:
                        dataclass_dict[d.dataclass] = 1

        self.dataclass_dict = dataclass_dict
        self.data_min, self.data_max, self.mean, self.std = [], [], [], []
        self.attrtype_dict = attrtype_dict
        self.datalist = datalist

        self.compute_data_min_max()
        self.encode()

    def encode(self):
        self.datalist = [Data(self.normalize(d), d.dataclass) for d in self.datalist]

    def decode(self, data):
        if isinstance(data, Data):
            res = copy.deepcopy(data)
            res.attrlist = self.denormalize(res.attrlist)
            return res
        else:
            return self.denormalize(data)

    def normalize(self, attrlist):
        # return [(num - num_min) / (num_max - num_min) if num_min != num_max else num for
        #         num, num_min, num_max in
        #         zip(attrlist, self.data_min, self.data_max)]
        # z-score
        return [(num - mean) / std if std != 0 else num for
                num, std, mean in
                zip(attrlist, self.std, self.mean)]

    def denormalize(self, data):
        # return [num * (data_max - data_min) + data_min for num, data_min, data_max in
        #         zip(data, self.data_min, self.data_max)]
        # z-score
        return [num * std + mean for num, std, mean in
                zip(data, self.std, self.mean)]

    def reverse_data(self):
        return [[d[i] for d in self.datalist] for i in range(self.datalist[0].__len__())]

    def compute_data_min_max(self):
        data_matrix = self.reverse_data()
        self.data_min = [min(d) for d in data_matrix]
        self.data_max = [max(d) for d in data_matrix]
        self.mean = [np.mean(d) for d in data_matrix]
        self.std = [np.std(d) for d in data_matrix]

    @property
    def get_original_datalist(self):
        return list(map(self.decode, self.datalist))

    def get_positive(self):
        return sorted(self.dataclass_dict.items(), key=lambda kv: (kv[1], kv[0]))[-1][0]

    def get_negative(self):
        if len(self.dataclass_dict) == 2:
            return sorted(self.dataclass_dict.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
        else:
            return [kv[0] for kv in sorted(self.dataclass_dict.items(), key=lambda kv: (kv[1], kv[0]))[:-1]]

    def __getitem__(self, index):
        data = self.datalist[index]
        return torch.Tensor(data.attrlist), data.dataclass

    def __len__(self):
        return self.datalist.__len__()


if __name__ == '__main__':
    dataset = MyDataSet('yeast5.dat', encode=False)
    print(dataset.datalist[0].attrlist)

    dataset = MyDataSet('yeast5.dat', encode=True)
    print(dataset.decode(dataset.datalist[0].attrlist))
    pass
