from enum import Enum

import copy


class DataType(Enum):
    CONTINUOUS = 0
    DISCRETE = 1
    TWO_VALUE = 2


str_to_num_dicts = []


def transform_str_to_num(s, i):
    if str_to_num_dicts.__len__() == i:
        str_to_num_dicts.append({})

    if s in str_to_num_dicts[i]:
        n = str_to_num_dicts[i][s]
    else:
        try:
            n = eval(s)
        except NameError:
            n = 0 if str_to_num_dicts[i].__len__() == 0 else list(str_to_num_dicts[i].values())[-1] + 1
            str_to_num_dicts[i][s] = n
    s = n

    return s


def transform_num_to_str(n, i):
    try:
        s = {v: k for k, v in str_to_num_dicts[i].items()}[n]
    except KeyError:
        s = n

    return s


class Data:

    def __init__(self, inputs, dataclass, sep=','):
        if isinstance(inputs, str):
            row_split = inputs.split(sep)
            self.dataclass = row_split.pop().split('.')[0].strip()
            self.attrlist = [transform_str_to_num(attr.strip(), i)
                             for i, attr in enumerate(row_split)]

        if isinstance(inputs, list):
            self.attrlist = inputs
            self.dataclass = dataclass

    def to_data(self, attrtype_dict, attrtype_list):
        result = copy.deepcopy(self)
        result.attrlist = [attr for attr, attrtype in zip(self.attrlist, list(attrtype_dict.values())) if
                           attrtype in attrtype_list]

        return result

    def to_list(self, attrtype_dict, attrtype_list=None):
        if attrtype_list:
            if not isinstance(attrtype_list, list):
                attrtype_list = [attrtype_list]
            return [d for d, v in zip(self.attrlist, attrtype_dict.values()) if v in attrtype_list]
        else:
            return self.attrlist

    def __getitem__(self, item):
        return self.attrlist[item]

    def __setitem__(self, key, value):
        self.attrlist[key] = value

    def __str__(self):
        return ('Data -> [' + self.__len__() * '{:>2.2f}, ' + '], Class -> {}').format(*self.attrlist, self.dataclass)

    def __len__(self):
        return len(self.attrlist)
