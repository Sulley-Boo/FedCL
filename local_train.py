import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
from options import args_parser
import copy
import math
from utils import losses


args = args_parser()

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

    def train(self, args, net, net_glob, net_prev, op_dict):
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4,
                                          amsgrad=True)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        criterion = torch.nn.CrossEntropyLoss().cuda()

        cos = torch.nn.CosineSimilarity(dim=-1)
        pdist = torch.nn.PairwiseDistance(2)

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
                label_batch = label_batch.long()
                # _, _, outputs = net(inputs)
                _, _, aug_outputs = net(ema_inputs)
                label_batch = torch.topk(label_batch, 1)[1].squeeze(1)
                # net_glob = net_glob.cuda()
                inputs.requires_grad = True
                label_batch.requires_grad = False

                #add Contrastive loss
                _, pro1, outputs = net(inputs)
                _, pro2, _ = net_glob(inputs)

                posi = cos(pro1,pro2)
                # posi = pdist(pro1, pro2)

                logits = posi.reshape(-1, 1)
                net_prev.cuda()
                _, pro3, _ = net_prev(inputs)
                nega = cos(pro1,pro3)
                # nega = pdist(pro1, pro3)

                logits = torch.cat((logits, nega.reshape(-1, 1)),dim=1)

                logits /= args.temperature
                labels = torch.zeros(inputs.size(0)).cuda().long()

                loss2 = args.mu * criterion(logits, labels)
                loss1 = criterion(outputs, label_batch)
                loss_total = loss1 + loss2

                loss = loss_total
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    batch_loss.append(loss.item())
                    self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            with torch.no_grad():
                epoch_loss.append(np.array(batch_loss).mean())
            print(epoch_loss)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())
