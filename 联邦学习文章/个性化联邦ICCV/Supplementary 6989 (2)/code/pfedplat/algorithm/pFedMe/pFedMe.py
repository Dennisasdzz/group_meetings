
import pfedplat as fp
import numpy as np
import torch
import copy
from torch.optim import Optimizer
import time


class pFedMe(fp.Algorithm):
    def __init__(self,
                 name='pFedMe',
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
                 beta=1.0,
                 lam=15.0,
                 *args,
                 **kwargs):

        if params is not None:
            beta = params['beta']
            lam = params['lam']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' beta' + str(beta) + ' lam' + str(lam)

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(i, copy.deepcopy(
                model), device, train_setting, metric_list, lam) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
        self.beta = beta
        self.exist_per_model = True

    def run(self):

        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            com_time_start = time.time()

            self.send_train_order(self.epochs)

            w_locals, _ = self.send_require_training_result()
            com_time_end = time.time()
            cal_time_start = time.time()

            self.aggregate(w_locals)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

    def aggregate(self, w_locals):
        w_global = self.model.state_dict()
        averaged_params = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] / \
                        self.online_client_num
                else:
                    averaged_params[k] += local_model_params[k] / \
                        self.online_client_num

            w_global[k] = (1 - self.beta) * w_global[k] + \
                self.beta * averaged_params[k]
        self.model.load_state_dict(w_global)


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.1, lam=15.0, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lam=lam, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, old_model_params):
        for group in self.param_groups:
            for p, old_weight in zip(group['params'], old_model_params):

                p.data = p.data - group['lr'] * (p.grad.data + group['lam'] * (
                    p.data - old_weight.data) + group['mu'] * p.data)


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 lam=15.0,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)

        self.optimizer = pFedMeOptimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, lam=lam)
        self.lam = lam
        self.per_model = copy.deepcopy(self.model)

    def train(self, epochs):

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        per_model_parameters = self.old_model.parameters()

        self.model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step(per_model_parameters)

                for theta_tilde_param, per_model_parameter in zip(self.model.parameters(), per_model_parameters):
                    per_model_parameter.data = per_model_parameter.data - self.lam * \
                        self.lr * (per_model_parameter.data -
                                   theta_tilde_param.data)

        self.model_loss = float(loss)

        for p, new_param in zip(self.model.parameters(), per_model_parameters):
            p.data = new_param.data
        self.per_model.clone_from(self.model)
