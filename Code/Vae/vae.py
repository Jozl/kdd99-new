import os

import csv
import random

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from Code.Dataset.data import Data, DataType
from Code.Dataset.dataset import MyDataSet
from Code.Vae.net import NetVAE

torch.autograd.set_detect_anomaly(True)


class Vae:
    def __init__(self, data_name, target_class, module_features, learning_rate, batch_size, log=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_class = target_class

        dataset = MyDataSet(data_name, target_class=target_class)
        self.dataset = dataset

        self.dataloader_train = DataLoader(dataset=dataset, batch_size=batch_size,
                                           shuffle=True, drop_last=False)

        self.net = NetVAE(module_features=[dataset.datalist[0].__len__()] + module_features)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.targetdata_list = []

        self.log = log
        if log:
            self.dir_path = 'KDD99_FAKE_{}'.format(dataset.dataname, )
            self.file_path = '/from_{}_gen_{}_label={}_'.format(dataset.__len__(), 0,
                                                                target_class)
            if not os.path.exists(self.dir_path):
                os.mkdir(self.dir_path)
            self.loss_log = open(self.dir_path + self.file_path + 'loss_log.txt', 'w')

    def __call__(self, epochs: int):
        for epoch in range(epochs):
            self.train(epoch)

        self.net.eval()

        if self.log:
            self.loss_log.close()
        return self

    def train(self, epoch):
        self.net.train()

        log_loss = 0
        for batch_index, (data_inputs, data_classes) in enumerate(self.dataloader_train):
            for data_input, data_class in zip(data_inputs, data_classes):
                if data_class == self.target_class:
                    self.targetdata_list.append(data_input)

            loss = self.loss_function(data_inputs, *self.net(data_inputs))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            log_loss += loss

        log_loss = log_loss / (batch_index * self.batch_size + self.batch_size)

        if self.log:
            self.loss_log.write('epoch = {:>2d}: total_loss: {:>4.7f}\n'
                                .format(epoch + 1, log_loss))

    def next(self):
        data_input_tensor = self.targetdata_list[random.randint(0, self.targetdata_list.__len__() - 1)]
        data_output_tensor, _, _ = self.net(data_input_tensor)
        data_output = self.dataset.decode(data_output_tensor.tolist())
        # data_output.append(self.target_class + '.')
        data_output = Data(data_output, dataclass=self.target_class)

        return data_output

    @staticmethod
    def loss_function(target_tensor, input_tensor, mean, log_var) -> (Tensor, Tensor):
        reconstruction_loss = torch.nn.CosineSimilarity()(input_tensor, target_tensor).sum()
        # reconstruction_loss = torch.nn.BCELoss(reduction='sum')(input_tensor, target_tensor)
        kl_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean ** 2)

        return reconstruction_loss + kl_divergence


def main():
    data_name = 'kdd99_new_multi.dat'
    target_class = 'warezclient'
    # target_num = 100

    learning_rate = 0.00064
    batch_size = 20
    module_features = [10, 8]
    training_round = 100

    generator = Vae(data_name, target_class, module_features, learning_rate, batch_size, log=True)(training_round)

    output = generator.next()
    print(output)


if __name__ == '__main__':
    main()
