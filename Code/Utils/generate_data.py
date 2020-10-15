from Code.Dataset.data import DataType
from Code.Dataset.dataset import MyDataSet
from Code.Smote.smote_cons import Smote
from Code.Vae.trainer import Trainer
from Code.Vae.vae import Vae


def gen_with_vae(target_class, target_num, data_name, ):
    learning_rate = 0.00064
    batch_size = 50
    module_features = [16,10]
    training_round = 100

    vae = Vae(data_name, target_class, module_features, learning_rate, batch_size, log=True)(training_round)
    return [vae.next() for _ in range(target_num)]


def gen_with_smote(target_class, target_num, data_name, ):
    smote = Smote(data_name, target_class, )
    return [smote.next() for _ in range(target_num)]


def gen_with_smote_enn(target_class, target_num, data_name, ):
    smote = Smote(data_name, target_class, under_sampling='enn')
    return [smote.next() for _ in range(target_num)]


def gen_with_smote_rsb(target_class, target_num, data_name, ):
    smote = Smote(data_name, target_class, under_sampling='rsb')
    return [smote.next() for _ in range(target_num)]
