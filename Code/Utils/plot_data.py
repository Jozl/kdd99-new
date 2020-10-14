import numpy as np

import matplotlib.pyplot as plt

from Code.Dataset.data import DataType
from Code.Dataset.dataset import MyDataSet
from Code.Utils.generate_data import gen_with_smote, gen_with_smote_enn, gen_with_vae, gen_with_smote_rsb


def draw(data, color, hist=True, plot=True):
    x = np.arange(np.min(data), np.max(data), 0.01)
    y = normfun(x, np.mean(data), np.std(data))

    if hist:
        # plt.hist(data, bins=64, rwidth=0.9, facecolor=color, alpha=0.5, density=True)
        plt.hist(data, bins=64, rwidth=0.9, facecolor=color, alpha=0.5)
    if plot:
        plt.plot(x, y, color=color)


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


if __name__ == '__main__':
    dataname = 'yeast5.dat'
    print(dataname)
    # dataname = 'kdd99_new_multi.dat'

    dataset = MyDataSet(dataname, encode=True)

    reversed_class_dict = dict(zip(dataset.dataclass_dict.values(), dataset.dataclass_dict.keys()))
    len_list = list(reversed_class_dict.keys())
    len_list.sort()

    positive_len, negative_len = len_list[1], len_list[0]
    positive_class, negative_class = reversed_class_dict[positive_len], reversed_class_dict[negative_len]
    negative_len = positive_len - negative_len

    dataset_negative = MyDataSet(dataname, target_class=negative_class, encode=True)
    data_rev_negative = dataset_negative.reverse_data()

    for expend_alg in [
        # None,
        'vae',
        'smote',
        'smote_enn',
        'smote_rsb',
    ]:
        for i, data_neg in enumerate( data_rev_negative):
            if list(dataset_negative.attrtype_dict.values())[i] not in [DataType.CONTINUOUS]:
                continue

            try:
                atr_name = 'Atr-{}'.format(i)

                data_predict = eval('gen_with_{}'.format(expend_alg))(negative_class, negative_len, dataname)
                # data_predict = [d.to_data(attrtype_dict=dataset.attrtype_dict, attrtype_list=[DataType.CONTINUOUS]) for d in
                #                 data_predict]
                data_predict = [d.attrlist[i] for d in data_predict]

                draw(data_predict, 'red')
                draw(data_neg, 'green')
                # draw(data, 'blue')

                data_neg.extend(data_predict)
                draw(data_neg, 'black', hist=False)

                title = '{}_{}_{}'.format(dataname, expend_alg, atr_name)

                plt.title(title)
                plt.xlabel('value')
                plt.ylabel('times')
                # 输出
                plt.savefig(title + '.png')
                plt.show()
                plt.close()
            except MemoryError:
                print('memory error as atr {}'.format(i))
