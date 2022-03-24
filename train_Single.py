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
from torch.utils.data import DataLoader
from local_train import LocalUpdate
from networks.models import ModelFedCon, DenseNet121



def test(epoch, save_mode_path):
    checkpoint_path = save_mode_path

    checkpoint = torch.load(checkpoint_path)
    # net = ModelFedCon(out_dim=args.out_dim, n_classes=7)
    net = DenseNet121(out_dim=args.out_dim,out_size=7,drop_rate=args.drop_rate)
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
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    AUROCs, Accus, Senss, Specs, _, F1 = epochVal_metrics_test(model, test_dataloader, thresh=0.4)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    F1_avg = np.array(F1).mean()

    return AUROC_avg, Accus_avg, Senss_avg, Specs_avg,F1_avg

snapshot_path = 'model/'

if __name__ == "__main__":
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

    train_dataset = DataLoader(train_dataset, batch_size=32,shuffle=True,num_workers=8)
    model = DenseNet121(out_dim=args.out_dim, out_size=7, drop_rate=args.drop_rate)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4,
                                 amsgrad=True)

    print('begin training')
    train_loss = []
    for epoch in range(200):
        model.train()
        for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(train_dataset):
            image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
            ema_inputs = ema_image_batch
            inputs = image_batch
            _, _, outputs = model(inputs)
            _, _, aug_outputs = model(ema_inputs)
            label_batch = torch.topk(label_batch, 1)[1].squeeze(1)
            loss = criterion(outputs,label_batch.long()) + criterion(aug_outputs,label_batch.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            print('epoch:',epoch,'loss:',loss.item())

        model.eval()
        with torch.no_grad():
            if (epoch + 1) % 10 == 0:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch + 1) + '.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }
                    , save_mode_path
                )
                AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg = test(epoch, save_mode_path)
                logging.info("\nTEST Student: Communication round: {}".format(epoch + 1))
                logging.info(
                    "\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, Test F1: {:6f}"
                        .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))