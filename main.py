import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import torch.nn.functional as F
from torch.nn import BCELoss
from torch.optim import Adam

from log import Mylogging
from load_dataset import get_dataloader, get_dataset
from model import DevignModel
from train import train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_data', type=str, help='Input Directory of the parser',
                        default='/home/survey_devign/explain_experiments')
    parser.add_argument('--pro_data_dir', type=str, help='Input Directory of the parser',
                        default='/home/Devign_PyG/dataset')
    parser.add_argument('--model_dir', type=str, help='Input Directory of the parser',
                        default='/home/Devign_PyG/models')
    parser.add_argument('--data_split_ratio', type=list, help='Input Directory of the parser',default=[0.8, 0.1, 0.1])
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=8)
    parser.add_argument('--max_edge_types', type=int, help='Batch Size for training', default=2)
    parser.add_argument('--epochs', type=int, help='Batch Size for training', default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    Mylogger = Mylogging()
    Mylogger.debug('-----Start-----')
    torch.manual_seed(22)
    args = parse_args()

    dataset = get_dataset(args)
    loader = get_dataloader(dataset, args)

    model = DevignModel(input_dim=args.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=args.max_edge_types)

    Mylogger.info('#' * 100)
    loss_function = F.cross_entropy
    optim = Adam(model.parameters(), lr=1e-4, weight_decay=1.3e-6)
    Mylogger.info(f'batch_size:{args.batch_size}')
    Mylogger.info('lr:1e-4')
    Mylogger.info('weight_decay:1.3e-6')
    train(model=model, dataset=loader, loss_function=loss_function, optimizer=optim, 
            args=args, save_path=args.model_dir + '/batch-' + str(args.batch_size), epochs=args.epochs) 
