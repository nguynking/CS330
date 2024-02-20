"""Implementation of prototypical networks for Metabolism."""
import sys
sys.path.append('..')
import argparse
import os

import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils import tensorboard

import omniglot
import util  # pylint: disable=unused-import

from submission.protonet import ProtoNet
from data_generator import NCI, Metabolism
from torch.utils.data import dataset, sampler, dataloader
from learner import FCNet

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600


def identity(x):
    return x

def main(args):

    print(args)

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/metabolism.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.lr_{args.learning_rate}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    protonet = ProtoNet(args.learning_rate, log_dir, DEVICE, args.compile, args.backend, learner=FCNet(args=args, x_dim=1024, hid_dim=500), val_interval=2000, save_interval=2000, bio=True)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        # Assign some equivalent names
        args.num_classes = args.num_way
        args.update_batch_size = args.num_support
        args.update_batch_size_eval = args.num_query
        args.metatrain_iterations = args.num_train_iterations
        args.meta_batch_size = args.batch_size

        dataloader_meta_train = Metabolism(args, 'train', return_list=True)
        # dataloader_meta_val = Metabolism(args, 'val', return_list=True)
        dataloader_meta_val = Metabolism(args, 'test', return_list=True)

        protonet.train(
            dataloader_meta_train,
            dataloader_meta_val,
            writer
        )

    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS,
            args.num_workers
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default=2, 
                        help=('needed to specify omniglot dataloader'))
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    
    parser.add_argument('--datadir', default='xxx', type=str, help='directory for datasets.')

    args = parser.parse_args()

    if args.cache == True:
        # Download Omniglot Dataset
        if not os.path.isdir("./omniglot_resized"):
            gdd.download_file_from_google_drive(
                file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
                dest_path="./omniglot_resized.zip",
                unzip=True,
            )
        assert os.path.isdir("./omniglot_resized")
    else:
        main(args)
