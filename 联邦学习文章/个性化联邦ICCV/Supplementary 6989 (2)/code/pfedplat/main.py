
import pfedplat as fp
import numpy as np
import argparse
import torch
import time
import sys
torch.multiprocessing.set_sharing_strategy('file_system')


def outFunc(alg):
    loss_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        training_loss = metric_history['training_loss'][-1]
        if training_loss is None:
            continue
        loss_list.append(training_loss)
    loss_list = np.array(loss_list)

    local_acc_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        local_acc_list.append(metric_history['test_accuracy'][-1])
    local_acc_list = np.array(local_acc_list)
    p = np.ones(len(local_acc_list))
    local_acc_fairness = np.arccos(
        local_acc_list @ p / (np.linalg.norm(local_acc_list) * np.linalg.norm(p)))

    simulated_acc_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        simulated_acc_list.append(
            metric_history['simulated_test_accuracy'][-1])
    simulated_acc_list = np.array(simulated_acc_list)
    simulated_acc_fairness = np.arccos(
        simulated_acc_list @ p / (np.linalg.norm(simulated_acc_list) * np.linalg.norm(p)))

    global_acc_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        global_acc_list.append(metric_history['global_test_accuracy'][-1])
    global_acc_list = np.array(global_acc_list)
    global_acc_fairness = np.arccos(
        global_acc_list @ p / (np.linalg.norm(global_acc_list) * np.linalg.norm(p)))
    if alg.exist_per_model:

        per_model_local_acc_list = []
        for i, per_model_metric_history in enumerate(alg.comm_log['client_per_model_metric_history']):
            per_model_local_acc_list.append(
                per_model_metric_history['test_accuracy'][-1])
        per_model_local_acc_list = np.array(per_model_local_acc_list)
        per_model_local_acc_fairness = np.arccos(
            local_acc_list @ p / (np.linalg.norm(local_acc_list) * np.linalg.norm(p)))

        per_model_simulated_acc_list = []
        for i, per_model_metric_history in enumerate(alg.comm_log['client_per_model_metric_history']):
            per_model_simulated_acc_list.append(
                per_model_metric_history['simulated_test_accuracy'][-1])
        per_model_simulated_acc_list = np.array(per_model_simulated_acc_list)
        per_model_simulated_acc_fairness = np.arccos(per_model_simulated_acc_list @ p / (
            np.linalg.norm(per_model_simulated_acc_list) * np.linalg.norm(p)))

        per_model_global_acc_list = []
        for i, per_model_metric_history in enumerate(alg.comm_log['client_per_model_metric_history']):
            per_model_global_acc_list.append(
                per_model_metric_history['global_test_accuracy'][-1])
        per_model_global_acc_list = np.array(per_model_global_acc_list)
        per_model_global_acc_fairness = np.arccos(
            per_model_global_acc_list @ p / (np.linalg.norm(per_model_global_acc_list) * np.linalg.norm(p)))

    stream_log = ""
    stream_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
    stream_log += 'round {}'.format(alg.current_comm_round) + \
        ' training_num {}'.format(alg.current_training_num) + '\n'
    stream_log += f'Mean Global Test loss: {format(np.mean(loss_list), ".6f")}' + \
        '\n' if len(loss_list) > 0 else ''
    stream_log += 'global model test: \n'
    stream_log += f'Global Test Acc: {format(np.mean(global_acc_list/100), ".3f")}({format(np.std(global_acc_list/100), ".3f")}), ave: {format(np.mean(global_acc_list), ".6f")}, std: {format(np.std(global_acc_list), ".6f")}, angle: {format(global_acc_fairness, ".6f")}, min: {format(np.min(global_acc_list), ".6f")}, max: {format(np.max(global_acc_list), ".6f")}' + '\n'
    if alg.exist_per_model:
        stream_log += 'per model test: \n'
        stream_log += f'Global Test Acc: {format(np.mean(per_model_global_acc_list/100), ".3f")}({format(np.std(per_model_global_acc_list/100), ".3f")}), ave: {format(np.mean(per_model_global_acc_list), ".6f")}, std: {format(np.std(per_model_global_acc_list), ".6f")}, angle: {format(per_model_global_acc_fairness, ".6f")}, min: {format(np.min(per_model_global_acc_list), ".6f")}, max: {format(np.max(per_model_global_acc_list), ".6f")}' + '\n'
        stream_log += f'Simulated Test Acc: {format(np.mean(per_model_simulated_acc_list/100), ".3f")}({format(np.std(per_model_simulated_acc_list/100), ".3f")}), ave: {format(np.mean(per_model_simulated_acc_list), ".6f")}, std: {format(np.std(per_model_simulated_acc_list), ".6f")}, angle: {format(per_model_simulated_acc_fairness, ".6f")}, min: {format(np.min(per_model_simulated_acc_list), ".6f")}, max: {format(np.max(per_model_simulated_acc_list), ".6f")}' + '\n'
        stream_log += f'Local Test Acc: {format(np.mean(per_model_local_acc_list/100), ".3f")}({format(np.std(per_model_local_acc_list/100), ".3f")}), ave: {format(np.mean(per_model_local_acc_list), ".6f")}, std: {format(np.std(per_model_local_acc_list), ".6f")}, angle: {format(per_model_local_acc_fairness, ".6f")}, min: {format(np.min(per_model_local_acc_list), ".6f")}, max: {format(np.max(per_model_local_acc_list), ".6f")}' + '\n'
    stream_log += '\n'
    alg.stream_log = stream_log + alg.stream_log
    print(stream_log)


def read_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='seed', type=int, default=1)

    parser.add_argument(
        '--device', help='device: -1, 0, 1, or ...', type=int, default=0)

    parser.add_argument('--model', help='model name;',
                        type=str, default='CNN_CIFAR10_FedAvg')

    parser.add_argument('--algorithm', help='algorithm name;',
                        type=str, default='FedAvg')

    parser.add_argument('--dataloader', help='dataloader name;',
                        type=str, default='DataLoader_cifar10_non_iid')

    parser.add_argument('--SN', help='split num', type=int, default=200)

    parser.add_argument('--PN', help='pick num', type=int, default=2)

    parser.add_argument('--B', help='batch size', type=int, default=50)

    parser.add_argument('--NC', help='client_class_num', type=int, default=2)

    parser.add_argument(
        '--balance', help='balance or not for pathological separation', type=str, default='True')

    parser.add_argument(
        '--Diralpha', help='alpha parameter for dirichlet', type=float, default=0.1)

    parser.add_argument('--types', help='dataloader label types;',
                        type=str, default='default_type')

    parser.add_argument('--N', help='client num', type=int, default=100)

    parser.add_argument(
        '--C', help='select client proportion', type=float, default=1.0)

    parser.add_argument('--R', help='communication round',
                        type=int, default=3000)

    parser.add_argument('--E', help='local epochs', type=int, default=1)

    parser.add_argument('--test_interval',
                        help='test interval', type=int, default=1)

    parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('--decay', help='learning rate decay',
                        type=float, default=0.999)
    parser.add_argument('--momentum', help='momentum', type=float, default=0.0)

    parser.add_argument('--theta', help='fairness angle',
                        type=float, default=11.25)
    parser.add_argument('--s', help='line search parameter',
                        type=int, default=1)

    parser.add_argument(
        '--gamma', help='fairness angle of FedPG', type=float, default=1.0)

    parser.add_argument('--alpha', help='alpha of APFL',
                        type=float, default=0.1)
    parser.add_argument(
        '--tau', help='parameter tau in FedRep', type=int, default=1)

    parser.add_argument(
        '--lam', help='parameter tau in Ditto/FedAMP/pFedMe/pFedGF', type=float, default=0.1)

    parser.add_argument(
        '--epsilon', help='parameter epsilon in FedMGDA+', type=float, default=0.1)

    parser.add_argument('--q', help='parameter q in qFedAvg',
                        type=float, default=0.1)

    parser.add_argument(
        '--eth', help='parameter epsilon in SplitGP', type=float, default=0.2)

    parser.add_argument('--t', help='parameter t in TERM', type=int, default=1)

    parser.add_argument(
        '--alphaK', help='parameter alphaK in FedAMP', type=float, default=5e-3)
    parser.add_argument(
        '--sigma', help='parameter sigma in FedAMP', type=float, default=0.1)

    parser.add_argument(
        '--mu', help='parameter mu in FedProx', type=float, default=0.0)

    parser.add_argument(
        '--beta', help='parameter K in pFedMe', type=float, default=1.0)

    parser.add_argument('--M', help='parameter M in FedEM',
                        type=int, default=3)

    parser.add_argument('--dishonest_num',
                        help='dishonest number', type=int, default=0)
    parser.add_argument('--scaled_update',
                        help='scaled update attack', type=str, default='None')
    parser.add_argument('--random_update',
                        help='random update attack', type=str, default='None')
    parser.add_argument(
        '--zero_update', help='zero update attack', type=str, default='None')

    parser.add_argument(
        '--sleep_sec', help='parameter K in pFedMe', type=int, default=0)

    try:
        parsed = vars(parser.parse_args())
        sleep_sec = parsed['sleep_sec']
        time.sleep(sleep_sec)
        return parsed
    except IOError as msg:
        parser.error(str(msg))


def initialize(params):
    fp.setup_seed(seed=params['seed'])
    device = torch.device(
        'cuda:' + str(params['device']) if torch.cuda.is_available() and params['device'] != -1 else "cpu")
    Model = getattr(sys.modules['pfedplat'], params['model'])
    model = Model(device)
    Dataloader = getattr(sys.modules['pfedplat'], params['dataloader'])
    data_loader = Dataloader(
        params=params, input_require_shape=model.input_require_shape)
    model.generate_net(data_loader.input_data_shape,
                       data_loader.target_class_num)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=params['lr'], momentum=params['momentum'])
    train_setting = {'criterion': torch.nn.CrossEntropyLoss(
    ), 'optimizer': optimizer, 'lr_decay': params['decay']}
    test_interval = params['test_interval']
    dishonest_num = params['dishonest_num']
    scaled_update = eval(params['scaled_update'])
    if scaled_update is not None:
        scaled_update = float(scaled_update)
    dishonest = {'dishonest_num': dishonest_num,
                 'scaled_update': scaled_update,
                 'random_update': eval(params['random_update']),
                 'zero_update': eval(params['zero_update'])}
    Algorithm = getattr(sys.modules['pfedplat'], params['algorithm'])
    algorithm = Algorithm(data_loader=data_loader,
                          model=model,
                          device=device,
                          train_setting=train_setting,
                          client_num=data_loader.pool_size,
                          online_client_num=int(
                              data_loader.pool_size * params['C']),
                          metric_list=[fp.Correct()],
                          max_comm_round=params['R'],
                          max_training_num=None,
                          epochs=params['E'],
                          outFunc=outFunc,
                          params=params,
                          dishonest=dishonest,
                          write_log=True)
    algorithm.test_interval = test_interval
    return data_loader, algorithm


if __name__ == '__main__':
    params = read_params()
    data_loader, algorithm = initialize(params)
    algorithm.run()
