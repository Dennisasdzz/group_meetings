
import pfedplat as fp
import numpy as np
import random
import torch
import copy
import json


class Algorithm:

    def __init__(self,
                 name='Algorithm',
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
                 *args,
                 **kwargs):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [fp.Client(i, copy.deepcopy(model), device, train_setting, metric_list) for i in range(client_num)]  
            data_loader.allocate(client_list)  
        elif client_num is None and client_list is None:
            raise RuntimeError('Both of client_num and client_list cannot be None or not None.')
        if online_client_num is None:
            online_client_num = client_num
        
        if dishonest is not None:
            dishonest_indices = np.random.choice(client_num, dishonest['dishonest_num'] ,replace=False).tolist()
            for idx, client in enumerate(client_list):
                if idx in dishonest_indices:
                    client.dishonest = dishonest
        
        choose_client_indices = list(np.random.choice(client_num, online_client_num, replace=False))
        self.online_client_list = [client_list[i] for i in choose_client_indices]
        if client_num > online_client_num:
            print(choose_client_indices)
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        if max_comm_round is None:
            max_comm_round = 10**10
        if max_training_num is None:
            max_training_num = 10**10
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.model = model
        self.train_setting = train_setting
        self.client_num = client_num
        self.client_list = client_list
        self.online_client_num = online_client_num
        self.max_comm_round = max_comm_round
        self.max_training_num = max_training_num
        self.epochs = epochs
        self.save_name = save_name
        self.outFunc = outFunc
        self.comm_trace = None
        self.current_comm_round = 0
        self.current_training_num = 0
        self.model.to(device)
        
        self.metric_list = metric_list
        self.write_log = write_log
        self.params = params
        self.dishonest = dishonest
        self.stream_log = ""
        self.save_folder=''
        
        self.comm_log = {'client_metric_history': [],  
                         'client_per_model_metric_history': [],  
                         'training_num': []}  
        
        self.lr = self.train_setting['optimizer'].defaults['lr']
        self.initial_lr = self.lr
        
        self.optimizer = train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.optimizer.defaults = train_setting['optimizer'].defaults
        
        self.result_model = None
        
        self.test_interval = 1
        
        self.exist_per_model = False
        
        self.communication_time = 0
        self.computation_time = 0

    def run(self):
        raise RuntimeError('error in Algorithm: This function must be rewritten in the child class. (该函数必须在子类中被重写！)')

    @staticmethod
    def update_learning_rate(optimizer, lr):
        optimizer.defaults['lr'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate(self):
        self.lr = self.initial_lr * self.train_setting['lr_decay']**self.current_comm_round
        self.update_learning_rate(self.optimizer, self.lr)  
        
        self.send_update_learning_rate_order()

    def update_model(self, model, optimizer, lr, g):
        self.update_learning_rate(optimizer, lr)
        for i, p in enumerate(model.parameters()):
            p.grad = g[model.Loc_reshape_list[i]]  
        optimizer.step()

    def terminated(self, update_count=False):
        
        self.adjust_learning_rate()
        
        self.send_sync_model(update_count, target_client_list=self.client_list)
        self.comm_log['training_num'].append(self.current_training_num)
        
        if self.current_comm_round % self.test_interval == 0:
            self.test(self.result_model)  
            
            if callable(self.outFunc):
                self.outFunc(self)
        if self.current_comm_round >= self.max_comm_round or self.current_training_num >= self.max_training_num:
            return True
        else:
            
            if self.online_client_num < self.client_num:
                choose_client_indices = list(np.random.choice(self.client_num, self.online_client_num, replace=False))
                print(choose_client_indices)
                self.online_client_list = [self.client_list[i] for i in choose_client_indices]
            return False

    def send_sync_model(self, update_count=True, model=None, target_client_list=None):
        
        if model is None:
            model = self.model
        if target_client_list is None:
            target_client_list = self.online_client_list  
        
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'sync', 'w_global': model.state_dict()}
            client.get_message(msg)
        if update_count:
            self.current_comm_round += 1  

    def send_update_learning_rate_order(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'update_learning_rate', 'current_comm_round': self.current_comm_round}
            client.get_message(msg)

    def send_cal_all_batches_loss_order(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'cal_all_batches_loss'}
            client.get_message(msg)

    def send_cal_all_batches_gradient_loss_order(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'cal_all_batches_gradient_loss'}
            client.get_message(msg)

    def send_train_order(self, epochs, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'train', 'epochs': epochs, 'lr': self.lr}
            client.get_message(msg)

    def send_test_order(self):
        
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'test'}
            client.get_message(msg)

    def send_require_cal_all_batches_loss_result(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        l_locals = []  
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_cal_all_batches_loss_result'}
            msg = client.get_message(msg)
            l_locals.append(msg['l_local'])
        
        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return l_locals

    def send_require_all_batches_gradient_loss_result(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        g_locals = []  
        l_locals = []  
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_all_batches_gradient_loss_result'}
            msg = client.get_message(msg)
            g_locals.append(msg['g_local'])
            l_locals.append(msg['l_local'])
        
        g_locals = torch.stack([g_locals[i] for i in range(len(g_locals))])
        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return g_locals, l_locals

    def send_require_client_model(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        
        m_locals = []
        l_locals = []
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_client_model', 'requires_grad': 'False'}
            msg = client.get_message(msg)
            m_locals.append(msg['m_local'])
            l_locals.append(msg['l_local'])
        return m_locals, l_locals

    def send_require_client_per_model(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        
        m_locals = []
        l_locals = []
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_client_per_model', 'requires_grad': 'False'}
            msg = client.get_message(msg)
            m_locals.append(msg['m_local'])
            l_locals.append(msg['l_local'])
        return m_locals, l_locals

    def send_require_training_result(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        
        w_locals = []
        l_locals = []
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_training_result'}
            msg = client.get_message(msg)
            w_locals.append(msg['w_local'])
            l_locals.append(msg['l_local'])
        return w_locals, l_locals

    def send_require_attr(self, attr='local_training_number', target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        
        attrs = []
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_attribute_value', 'attr': attr}
            msg = client.get_message(msg)
            attrs.append(msg['attr'])
        return attrs

    def test(self, result_model=None):
        if result_model is not None:
            self.send_sync_model(update_count=False, model=result_model, target_client_list=self.client_list)  
        
        self.send_test_order()
        
        self.comm_log['client_metric_history'] = []
        self.comm_log['client_per_model_metric_history'] = []
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_test_result'}
            msg = client.get_message(msg)  
            self.comm_log['client_metric_history'].append(msg['metric_history'])
            self.comm_log['client_per_model_metric_history'].append(msg['per_model_metric_history'])
        
        if self.write_log:
            self.save_log()

    def save_log(self):
        
        save_dict = {'algorithm name': self.name}
        save_dict['client num'] = self.client_num
        save_dict['communication round'] = self.current_comm_round
        save_dict['epochs'] = self.epochs
        save_dict['communication log'] = self.comm_log
        save_dict['info'] = 'data loader name_' + self.data_loader.name + '_model name_' + self.model.name + '_train setting_' + str(self.train_setting) + '_client num_' + str(self.client_num) + '_max comm round_' + str(self.max_comm_round) + '_epochs_' + str(self.epochs)
        file_name = self.save_folder + self.save_name + '.json'
        fileObject = open(file_name, 'w')
        fileObject.write(json.dumps(save_dict))
        fileObject.close()
        file_name = self.save_folder + 'log_' + self.save_name + '.log'
        fileObject = open(file_name, 'w')
        fileObject.write(self.stream_log)
        fileObject.close()
