
import pfedplat as fp
import numpy as np
import torch
import copy
from torch.autograd import Variable
import time


class SplitGP_g(fp.Algorithm):
    def __init__(self,
                 name='SplitGP_g',
                 data_loader=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 params=None,
                 gamma=0.2,
                 eth=0.2,
                 *args,
                 **kwargs):
        if params is not None:
            gamma = params['gamma']
            eth = params['eth']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(
                train_setting['lr_decay']) + ' gamma' + str(gamma) + ' eth' + str(eth)

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(i, copy.deepcopy(
                model), device, train_setting, metric_list, gamma, eth) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
        self.gamma = gamma
        self.eth = eth
        self.exist_per_model = True

    def run(self):

        training_nums = torch.Tensor(self.send_require_attr(
            'local_training_number')).float().to(self.device)
        sums = torch.sum(training_nums)
        weights = training_nums / sums

        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            com_time_start = time.time()

            self.send_train_order(self.epochs)

            m_locals, _ = self.send_require_client_model()

            p_locals, _ = self.send_require_client_per_model()
            com_time_end = time.time()
            cal_time_start = time.time()

            self.aggregate(m_locals, p_locals, weights)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

    def aggregate(self, m_locals, p_locals, weights):

        params_mat = torch.stack(
            [m_local.span_model_params_to_vec() for m_local in m_locals])
        aggregate_params = weights @ params_mat
        self.model.reshape_vec_to_model_params(aggregate_params)

        per_params = [p_local.span_model_params_to_vec()
                      for p_local in p_locals]
        aggregate_per_params = weights @ torch.stack(per_params)
        for i, p_local in enumerate(p_locals):
            p_local.reshape_vec_to_model_params(
                self.gamma * per_params[i] + (1 - self.gamma) * aggregate_per_params)

            self.online_client_list[i].per_model.clone_from(p_local)


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 gamma=0.2,
                 eth=0.2,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)
        self.gamma = gamma
        self.eth = eth

        self.per_model = copy.deepcopy(model)
        self.per_optimizer = train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.per_model.parameters()), lr=self.lr)
        self.per_optimizer.defaults = copy.deepcopy(
            train_setting['optimizer'].defaults)

        self.test_per_model = copy.deepcopy(model)

    def train(self, epochs):

        model_predictor = copy.deepcopy(self.model.predictor)
        self.model = copy.deepcopy(self.per_model)
        self.model.predictor = model_predictor
        self.optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                p_out = self.per_model(batch_x)
                p_loss = self.criterion(p_out, batch_y)
                g_out = self.model(batch_x)
                g_loss = self.criterion(g_out, batch_y)
                loss = self.gamma * p_loss + (1 - self.gamma) * g_loss

                self.per_model.zero_grad()
                self.model.zero_grad()
                loss.backward()
                self.per_optimizer.step()
                self.optimizer.step()

        self.model_loss = float(loss)

        self.test_per_model = copy.deepcopy(self.model)
