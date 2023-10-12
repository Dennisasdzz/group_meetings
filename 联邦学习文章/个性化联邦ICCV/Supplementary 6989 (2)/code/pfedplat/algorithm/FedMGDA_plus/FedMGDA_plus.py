
import pfedplat as fp
import torch
import numpy as np
import copy
from pfedplat.algorithm.common.utils import get_d_mgdaplus_d
import time


class FedMGDA_plus(fp.Algorithm):
    def __init__(self,
                 name='FedMGDA+',
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
                 epsilon=0.1,
                 *args,
                 **kwargs):

        if params is not None:
            epsilon = params['epsilon']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' epsilon' + str(epsilon)

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(i, copy.deepcopy(
                model), device, train_setting, metric_list) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
        self.epsilon = epsilon

        self.comm_log['d_optimality_history'] = []
        self.comm_log['d_descent_history'] = []
        self.exist_per_model = True

    def run(self):

        batch_num = np.mean(self.send_require_attr('training_batch_num'))

        while not self.terminated(update_count=True):
            com_time_start = time.time()
            self.model.train()

            self.send_train_order(self.epochs)

            m_locals, l_locals = self.send_require_client_model()
            com_time_end = time.time()
            cal_time_start = time.time()

            g_locals = []
            old_models = self.model.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                grad = old_models - m_locals[idx].span_model_params_to_vec()
                g_locals.append(grad)
            g_locals = torch.stack(g_locals)
            g_locals /= torch.norm(g_locals, dim=1).reshape(-1, 1)
            training_nums = self.send_require_attr('local_training_number')
            lambda0 = np.array(training_nums) / sum(training_nums)

            d, d_optimal_flag, d_descent_flag = get_d_mgdaplus_d(
                g_locals, self.device, self.epsilon, lambda0)

            for i, p in enumerate(self.model.parameters()):
                p.grad = d[self.model.Loc_reshape_list[i]]
            self.optimizer.step()

            self.current_training_num += self.epochs * batch_num

            self.comm_log['d_optimality_history'].append(d_optimal_flag)
            self.comm_log['d_descent_history'].append(d_descent_flag)

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)
        self.per_model = copy.deepcopy(self.model)

    def train(self, epochs):

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        loss = self.cal_all_batches_loss(self.model)

        self.model_loss = loss

        self.model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.per_model.clone_from(self.model)
