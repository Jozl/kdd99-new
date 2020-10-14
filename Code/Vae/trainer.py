import os

import csv

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from Code.Dataset.data import Data, DataType
from Code.Dataset.dataset import MyDataSet
from Code.Vae.net import NetVAE


class Trainer:
    def __init__(self, dataset: MyDataSet, output_data_label, output_data_size, batch_size, learning_rate,
                 module_features, log=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_data_label = output_data_label
        self.output_data_size = output_data_size

        self.dataset = dataset
        self.dataloader_train = DataLoader(dataset=dataset, batch_size=batch_size,
                                           shuffle=True, drop_last=False)

        self.net = NetVAE(module_features=module_features)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)

        self.log = log
        if log:
            self.dir_path = 'KDD99_FAKE_{}'.format(dataset.data_name, )
            self.file_path = '/from_{}_gen_{}_label={}_'.format(dataset.__len__(), output_data_size,
                                                                self.output_data_label)
            if not os.path.exists(self.dir_path):
                os.mkdir(self.dir_path)
            self.loss_log = open(self.dir_path + self.file_path + 'loss_log.txt', 'w')
        self.output_data_list = []
        self._output_data = []

    def __call__(self, epochs: int):
        self.recon_loss_list = []
        self.kl_loss_list = []
        self.loss_list = []

        for epoch in range(epochs):
            self.train(epoch)

        self.net.eval()
        self.predict()

        if self.log:
            plt.plot(self.kl_loss_list, label='kl')
            plt.plot(self.recon_loss_list, label='recon')
            plt.plot(self.loss_list, label='sum')
            plt.xlabel('epoch')
            plt.legend()
            plt.savefig(self.dir_path + self.file_path + 'label={}_loss plt.jpg'.format(self.output_data_label))
            plt.close('all')

            self.loss_log.close()
            # os.startfile(self.dir_path)
        return self

    @property
    def output_data(self):
        return self._output_data

    def train(self, epoch):
        self.net.train()

        batch_index = 0
        recon_loss, kl_loss = 0, 0
        for batch_index, (inputs, labels) in enumerate(self.dataloader_train):
            for input_, label in zip(inputs, labels):
                if label == self.output_data_label:
                    self.output_data_list.append(input_)
            outputs, recon_loss, kl_loss = self.batch_op(inputs, batch_index, recon_loss, kl_loss)

        batch_index += 1
        recon_loss /= (batch_index * self.batch_size)
        kl_loss /= (batch_index * self.batch_size)

        if self.log:
            self.kl_loss_list.append(kl_loss)
            self.recon_loss_list.append(recon_loss)
            self.loss_list.append(recon_loss + kl_loss)
            self.loss_log.write('epoch = {:>2d}: recon_loss: {:>4.7f}, KL_loss: {:>4.7f}, total_loss: {:>4.7f}\n'
                                .format(epoch + 1, recon_loss, kl_loss,
                                        recon_loss + kl_loss))

    def predict(self):
        target_len = self.output_data_size
        data_inputs = self.output_data_list
        while len(data_inputs) < target_len:
            if len(data_inputs) == 0:
                print('no such data loaded!')
                return
            data_inputs.extend(data_inputs)
        data_inputs = data_inputs[:target_len]

        self.output_data_list.clear()
        for batch_index, data_input in zip(range(len(data_inputs)), data_inputs):
            data_output_net, _, _ = self.net(data_input)
            data_output = self.dataset.decode(data_output_net.tolist())
            data_output.append(self.output_data_label + '.')
            data_output = Data(data_output, attr_dict=self.dataset.attr_dict, data_class=self.output_data_label,
                               data_type=DataType.CONTINUOUS)
            self._output_data.append(data_output)

    def batch_op(self, input_, batch_index, recon_loss, kl_loss):
        assert isinstance(input_, Tensor)

        output, mean, log_var = self.net(input_)
        batch_recon_loss, batch_kl_loss = self.loss_function(input_tensor=output, target_tensor=input_,
                                                             mean=mean,
                                                             log_var=log_var)

        # self.loss_log.write('batch: {}, recon_loss: {:>4.7f}, KL_loss: {:>4.7f}, total_loss: {:>4.7f}\n'
        #                     .format(batch_index + 1, batch_recon_loss / self.batch_size,
        #                             batch_kl_loss / self.batch_size,
        #                             (batch_recon_loss + batch_kl_loss) / self.batch_size))

        recon_loss += batch_recon_loss
        kl_loss += batch_kl_loss
        batch_loss = recon_loss + kl_loss

        self.optimizer.zero_grad()
        batch_loss.backward(retain_graph=True)
        self.optimizer.step()

        return output, recon_loss, kl_loss

    @staticmethod
    def loss_function(input_tensor, target_tensor, mean, log_var) -> (Tensor, Tensor):
        reconstruction_loss = torch.nn.CosineSimilarity()(input_tensor, target_tensor).sum()
        # reconstruction_loss = torch.nn.BCELoss(reduction='sum')(input_tensor, target_tensor)
        kl_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean ** 2)

        return reconstruction_loss, kl_divergence


if __name__ == '__main__':
    target_class = 'land'
    data_name = 'kdd99_{}.kdd99'.format(target_class)

    dataset = MyDataSet(data_name, target_class=target_class, encode=True)
    trainer = Trainer(module_features=(dataset.single_continuous_data_len, 30, 20, 16), learning_rate=0.000918,
                      batch_size=100,
                      dataset=dataset, output_data_label=target_class, output_data_size=40)
    trainer(50)

    # Todo print
    print(trainer.output_data)

    dataset = MyDataSet(trainer.output_data,
                        target_class=target_class, encode=True)
    trainer = Trainer(module_features=(dataset.single_continuous_data_len, 30, 20, 16), learning_rate=0.000918,
                      batch_size=100,
                      dataset=dataset, output_data_label=target_class, output_data_size=80)
    trainer(50)
    # TODO print
    print('all over')
