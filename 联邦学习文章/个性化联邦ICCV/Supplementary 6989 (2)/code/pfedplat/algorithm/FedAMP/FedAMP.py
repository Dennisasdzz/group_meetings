
import pfedplat as fp
import numpy as np
import copy
import torch
from torch.autograd import Variable
import time


class FedAMP(fp.Algorithm):
    def __init__(self,
                 name='FedAMP',
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
                 alphaK=5e-3,
                 sigma=0.1,
                 lam=5e-7,
                 *args,
                 **kwargs):

        if params is not None:
            alphaK = params['alphaK']
            sigma = params['sigma']
            lam = params['lam']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(
                train_setting['lr_decay']) + ' alphaK' + str(alphaK) + ' sigma' + str(sigma) + ' lam' + str(lam)

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(i, copy.deepcopy(model), device, train_setting, metric_list,
                                  data_loader.input_data_shape, data_loader.target_class_num, alphaK, lam) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
        self.exist_per_model = True
        self.alphaK = alphaK
        self.sigma = sigma
        self.lam = lam

    def run(self):

        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            com_time_start = time.time()
            m_locals, _ = self.send_require_client_per_model()
            com_time_end = time.time()
            cal_time_start = time.time()
            u_list = self.cal_u(m_locals)
            self.aggregate(m_locals)

            self.send_u(u_list)

            self.send_train_order(self.epochs)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

    def aggregate(self, m_locals):
        w_locals = [m_local.state_dict() for m_local in m_locals]
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
        self.model.load_state_dict(averaged_params)

    def send_u(self, u_list):
        for idx, client in enumerate(self.online_client_list):
            msg = {'command': 'update_u', 'u': u_list[idx]}
            client.get_message_u(msg)

    def e(self, x):
        return torch.exp(-x / self.sigma) / self.sigma

    def cal_u(self, m_locals):

        params_mat = []
        for m_local_i in m_locals:
            params_mat.append(m_local_i.span_model_params_to_vec())
        params_mat = torch.stack(params_mat)

        u_list = []
        for i in range(self.online_client_num):

            coef = torch.zeros(self.online_client_num).float().to(self.device)
            weights_i_vec = params_mat[i, :]
            for j in range(self.online_client_num):
                if i != j:
                    weights_j_vec = params_mat[j, :]
                    sub = weights_i_vec - weights_j_vec
                    coef[j] = self.alphaK * self.e(sub @ sub)
                else:
                    coef[j] = 0.0
            coef = torch.sum(coef)

            xi_vec = torch.ones(len(m_locals)).float().to(self.device) * coef
            xi_vec[i] = 1 - coef

            u = xi_vec @ params_mat
            u_list.append(u)
        return u_list


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 input_data_shape=None,
                 target_class_num=None,
                 alphaK=5e-3,
                 lam=5e-7,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)

        self.per_model = copy.deepcopy(model)
        self.per_optimizer = train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.per_model.parameters()), lr=self.lr)
        self.per_optimizer.defaults = copy.deepcopy(
            train_setting['optimizer'].defaults)

        self.alphaK = alphaK
        self.lam = lam
        self.u = copy.deepcopy(model)

    def get_message_u(self, msg):
        if msg['command'] == 'update_u':
            u_params_vec = msg['u']
            self.u.reshape_vec_to_model_params(u_params_vec)

    def train(self, epochs):

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')
        self.old_per_model = copy.deepcopy(self.per_model)

        self.update_learning_rate(self.per_optimizer, self.lr)
        self.per_model.train()

        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.per_model(batch_x)
                loss = self.criterion(out, batch_y)

                model_params_vec = self.per_model.span_model_params_to_vec()
                u_params_vec = self.u.span_model_params_to_vec()
                sub = model_params_vec - u_params_vec
                loss += self.lam / (self.alphaK * 2) * sub @ sub

                self.per_model.zero_grad()
                loss.backward()
                self.per_optimizer.step()

        self.model_loss = float(loss)
