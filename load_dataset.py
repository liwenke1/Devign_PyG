import os
import json
import torch
from log import Mylogging
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch.utils.data import random_split

Mylogger = Mylogging()

def dump_txt(dataset, filename):
    with open(filename, 'w') as f:
        for data in dataset:
            f.write(data.name + '\n')


def read_json(filename):
    with open(filename,'r') as f:
        file = json.load(f)

    x = torch.tensor(file['node_features'],dtype=torch.float)

    edge_index_list = []
    for edge in file['graph']:
        edge_index_list.append([edge[0],edge[2]])
    edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()
    
    edge_attr_list = []
    for edge in file['graph']:
        edge_attr_list.append([edge[1]])
    edge_attr = torch.tensor(edge_attr_list,dtype=torch.float)

    y=torch.tensor([file['target']],dtype=torch.float)
    
    data=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,name=filename)
    return data


def get_dataset(args):
    Mylogger.info('Getting Dataset')
    return Devign(root=args.pro_data_dir, raw_data_path=args.raw_data)


def get_dataloader(dataset, args):
    Mylogger.info('Getting Dataloader')

    num_train = int(len(dataset) * args.data_split_ratio[0])
    num_vaild = int(len(dataset) * args.data_split_ratio[1])
    num_test = len(dataset) - num_vaild - num_train

    train, vaild, test = random_split(dataset, [num_train, num_vaild, num_test])

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dataloader['vaild'] = DataLoader(vaild, batch_size=args.batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    dump_txt(dataloader['train'].dataset, os.path.join(args.raw_data, 'train.txt'))
    dump_txt(dataloader['vaild'].dataset, os.path.join(args.raw_data, 'vaild.txt'))
    dump_txt(dataloader['test'].dataset, os.path.join(args.raw_data, 'test.txt'))
    return dataloader

class Devign(InMemoryDataset):

    def __init__(self, raw_data_path, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.raw_data_path = raw_data_path
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'Devign.pt']

    def process(self):
        data_list = []
        with open('/home/survey_devign/explain_experiments/all_files.txt','r') as f:
            dataset_path = f.readlines()
        for path in dataset_path:
            data = read_json(path.strip())
            if(data.num_nodes >= 15):
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])