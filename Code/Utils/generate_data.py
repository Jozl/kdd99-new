from Code.Dataset.data import DataType
from Code.Dataset.dataset import MyDataSet
from Code.Smote.smote_cons import Smote
from Code.Vae.trainer import Trainer
from Code.Vae.vae import Vae


def gen_with_vae(target_class, target_num, data_name, encode=True):
    learning_rate = 0.00064
    batch_size = 50
    training_round = 60

    vae = Vae(data_name, target_class, learning_rate, batch_size, encode=encode, log=True)(training_round)
    return [vae.next() for _ in range(target_num)]


def gen_with_smote(target_class, target_num, data_name, encode=True):
    smote = Smote(data_name, target_class, encode=encode)
    return [smote.next() for _ in range(target_num)]


def gen_with_smote_enn(target_class, target_num, data_name, encode=True):
    smote = Smote(data_name, target_class, encode=encode, under_sampling='enn')
    return [smote.next() for _ in range(target_num)]


def gen_with_smote_rsb(target_class, target_num, data_name, encode=True):
    smote = Smote(data_name, target_class, encode=encode, under_sampling='rsb')
    return [smote.next() for _ in range(target_num)]
