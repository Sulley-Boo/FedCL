from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
from FedAvg import FedAvg
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import *
from dataloaders import dataset
from torch.utils.data import DataLoader, Dataset
from utils import losses


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, image, label = self.dataset[self.idxs[item]]
        return items, index, image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)

        self.epoch = 0
        self.iter_num = 0
        self.base_lr = args.base_lr

    def train(self, args, net, op_dict):
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4,
                                          amsgrad=True)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        # loss_fn = losses.LabelSmoothingCrossEntropy()
        loss_fn = nn.CrossEntropyLoss().cuda()
        global_weight_collector = list(net_glob.cuda().parameters())

        # train and update
        epoch_loss = []
        print('begin training')
        for epoch in range(args.local_ep):
            batch_loss = []
            iter_max = len(self.ldr_train)
            print(iter_max)
            for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
                ema_inputs = ema_image_batch
                inputs = image_batch
                inputs.requires_grad = True
                label_batch.requires_grad = False
                _, activations, outputs = net(inputs)
                _,_, aug_outputs = net(ema_inputs)
                label_batch = torch.topk(label_batch, 1)[1].squeeze(1)

                loss = loss_fn(outputs, label_batch.long()) + loss_fn(aug_outputs, label_batch.long())

                #########################we implement FedProx Here###########################
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
                ##############################################################################

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            print(epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())




def split(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    print(len(dataset))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def test(epoch, save_mode_path):
    checkpoint_path = save_mode_path

    checkpoint = torch.load(checkpoint_path)
    net = DenseNet121(args.out_dim, out_size=7,drop_rate=args.drop_rate)

    model = net.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    test_dataset = dataset.CheXpertDataset(root_dir=args.test_path,
                                           csv_file=args.csv_file_test,
                                           transform=transforms.Compose([
                                               transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               normalize,
                                           ]))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    AUROCs, Accus, Senss, Specs, _, F1 = epochVal_metrics_test(model, test_dataloader, thresh=0.4)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    F1_avg = np.array(F1).mean()

    return AUROC_avg, Accus_avg, Senss_avg, Specs_avg,F1_avg


snapshot_path = 'model/'

AUROCs = []
Accus = []
Senss = []
Specs = []

user_id = [0,1,2,3,4,5,6,7,8,9]

flag_create = False
print('done')


if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.basicConfig(filename="log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = dataset.CheXpertDataset(root_dir=args.train_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])))
    # train_dataset, _ = torch.utils.data.random_split(train_dataset, [17503, 2500])

    dict_users = split(train_dataset, args.num_users)
    net_glob = DenseNet121(args.out_dim, out_size=7,drop_rate=args.drop_rate)

    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[0, 1])

    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    optim_locals = []

    for i in user_id:
        trainer_locals.append(LocalUpdate(args, train_dataset, dict_users[i]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())
        # optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr,betas=(0.9, 0.999), weight_decay=5e-4)#adam
        # optimizer = torch.optim.SGD(net_locals[i].parameters(), lr=args.base_lr,momentum=0.9, weight_decay=5e-4)#sgd
        optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4,
                                     amsgrad=True)  # amsgrad

        optim_locals.append(copy.deepcopy(optimizer.state_dict()))


    for com_round in range(args.rounds):
        print("begin")
        loss_locals = []

        for idx in user_id:
            if com_round * args.local_ep > 20:
                trainer_locals[idx].base_lr = 3e-4
            local = trainer_locals[idx]

            optimizer = optim_locals[idx]
            w, loss, op = local.train(args, net_locals[idx], optimizer)
            w_locals[idx] = copy.deepcopy(w)
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))

        with torch.no_grad():
            w_glob = FedAvg(w_locals)

        net_glob.load_state_dict(w_glob)

        for i in user_id:
            net_locals[i].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print(loss_avg, com_round)
        logging.info('Loss Avg {} Round {} LR {} '.format(loss_avg, com_round, args.base_lr))
        if (com_round + 1) % 10 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(com_round + 1) + '.pth')
            torch.save({
                'state_dict': net_glob.state_dict(),
            }
                , save_mode_path
            )
            AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg = test(com_round, save_mode_path)
            logging.info("\nTEST Student: Communication round: {}".format(com_round + 1))
            logging.info(
                "\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, Test F1: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))