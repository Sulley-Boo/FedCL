import argparse


def args_parser():
     parser = argparse.ArgumentParser()
     parser.add_argument('--model', type=str, default='densenet121', help='neural network used in training')
     # parser.add_argument('--train_path', type=str, default='Coronahack/task_train', help='dataset train dir') #dataset/task_3_train
     # parser.add_argument('--validation_path', type=str, default='Coronahack/task_validation', help='dataset validation dir') #dataset/task_3_validation
     # parser.add_argument('--test_path', type=str, default='Coronahack/task_test', help='dataset test dir') #dataset/task_3_test
     # parser.add_argument('--csv_file_train', type=str, default='Coronahack/training.csv', help='training set csv file') #dataset/training.csv
     # parser.add_argument('--csv_file_val', type=str, default='Coronahack/validation.csv', help='validation set csv file') #dataset/validation.csv
     # parser.add_argument('--csv_file_test', type=str, default='Coronahack/testing.csv', help='testing set csv file') #dataset/testing.csv

     parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))

     parser.add_argument('--train_path', type=str, default='dataset/task_3_train',
                         help='dataset train dir')
     parser.add_argument('--validation_path', type=str, default='dataset/task_3_validation',
                         help='dataset validation dir')
     parser.add_argument('--test_path', type=str, default='dataset/task_3_test',
                         help='dataset test dir')
     parser.add_argument('--csv_file_train', type=str, default='dataset/training.csv',
                         help='training set csv file')
     parser.add_argument('--csv_file_val', type=str, default='dataset/validation.csv',
                         help='validation set csv file')
     parser.add_argument('--csv_file_test', type=str, default='dataset/testing.csv',
                         help='testing set csv file')

     parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
     parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
     parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
     parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
     parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
     parser.add_argument('--model_buffer_size', type=int, default=1,
                         help='store how many previous models for contrastive loss')
     parser.add_argument('--seed', type=int,  default=1337, help='random seed')
     parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
     parser.add_argument('--local_ep', type=int,  default=1, help='local epoch')
     parser.add_argument('--num_users', type=int,  default=1, help='numbers of users')
     parser.add_argument('--rounds', type=int,  default=200, help='communication rounds')
     parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
     parser.add_argument('--mu', type=float, default=1, help='the mu parameter for Contrastive loss')
     parser.add_argument('--temperature', type=float, default=0.5,
                         help='the temperature parameter for contrastive loss')
     parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
     parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
     parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')


     ### tune
     parser.add_argument('--resume', type=str,  default=None, help='model to resume')
     parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
     parser.add_argument('--global_step', type=int,  default=0, help='global_step')
     ### costs
     parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
     parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
     parser.add_argument('--consistency', type=float,  default=1, help='consistency')
     parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
     parser.add_argument('--use_project_head', type=int, default=1)
     parser.add_argument('--sample_fraction', type=float, default=1.0,
                         help='how many clients are sampled in each round')
     args = parser.parse_args()
     return args
