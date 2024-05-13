import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np


class ChainNetDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        y1 = self.data_list[idx]['perform_respt']
        y2 = self.data_list[idx]['perform_tput']
        x = self.data_list[idx]
        return x, y1, y2

def ProcessData(root, numSamples):
    data_list = []
    
    f_arr = open(osp.join(root, f'raw/arrival.txt'))
    f_capa = open(osp.join(root, f'raw/capacityRatio.txt'))
    f_nc = open(osp.join(root, f'raw/numClasses.txt'))
    f_nn = open(osp.join(root, f'raw/numNodes.txt'))
    f_path = open(osp.join(root, f'raw/path.txt'))
    f_ser = open(osp.join(root, f'raw/serRatio.txt'))
    f_serProp = open(osp.join(root, f'raw/serStatProp.txt'))
    f_classcapa = open(osp.join(root, f'raw/classCapacityRatio.txt'))
    f_respt = open(osp.join(root, f'raw/resptRatio.txt'))
    f_tput = open(osp.join(root, f'raw/tputRatio.txt'))
    for idx in range(numSamples):
        row_capa = f_capa.readline().strip().split()
        row_nc = f_nc.readline().strip().split()
        row_nn = f_nn.readline().strip().split()

        row_capa = [float(w) for w in row_capa]
        row_nc = [int(w) for w in row_nc]
        row_nn = [int(w) for w in row_nn]

        len_eachpath = []
        edge = []
        fragTodev_edge = []
        fragnode = row_nn[0]+1
        realnode_feat = []
        hypernode_feat = []
        row = []
        col = []
        path_ref = []
        perform_respt = []
        perform_tput = []

        for ni in range(row_nn[0]):
            feat_vec = [0] * 5 
            feat_vec[0] = row_capa[ni]
            realnode_feat.append(feat_vec)
        
        max_num_frags = 20
        paths = np.zeros((row_nc[0], max_num_frags*2), dtype=int) 

        for cla in range(row_nc[0]):
            row_arr = f_arr.readline().strip().split()
            row_path = f_path.readline().strip().split()
            row_ser = f_ser.readline().strip().split()
            row_serProp = f_serProp.readline().strip().split()
            row_classcapa = f_classcapa.readline().strip().split()
            row_respt = f_respt.readline().strip().split()
            row_tput = f_tput.readline().strip().split()

            row_arr = [float(w) for w in row_arr]
            row_path = [int(w) for w in row_path]
            row_ser = [float(w) for w in row_ser]
            row_serProp = [float(w) for w in row_serProp]
            row_classcapa = [float(w) for w in row_classcapa]
            row_respt = [float(w) for w in row_respt]
            row_tput = [float(w) for w in row_tput]

            len_eachpath.append(2*len(row_path))

            hypernode_feat.append([1])

            perform_respt.append(row_respt[0])
            perform_tput.append(row_tput[0])

            row.extend([cla for _ in range(2*len(row_path))])
            col.extend([i for i in range(2*len(row_path))])

            # edge_idx 
            edge.append([fragnode, row_path[0]])
            fragTodev_edge.append([fragnode, row_path[0]])
            paths[cla][0] = fragnode
            paths[cla][1] = row_path[0]
            path_ref.extend([fragnode, row_path[0]])
            fragnode = fragnode+1

            feat_vec = [0] * 5 
            feat_vec[1] = row_ser[0]
            feat_vec[2] = row_serProp[0]
            feat_vec[3] = row_classcapa[0]
            realnode_feat.append(feat_vec)

            for ei in range(len(row_path)-1):
                edge.append([row_path[ei],fragnode])
                edge.append([fragnode,row_path[ei+1]])
                fragTodev_edge.append([fragnode, row_path[ei+1]])
                paths[cla][2*(ei+1)] = fragnode
                paths[cla][2*(ei+1)+1] = row_path[ei+1]
                path_ref.extend([fragnode, row_path[ei+1]])
                fragnode = fragnode+1

                feat_vec = [0] * 5 
                feat_vec[1] = row_ser[ei+1]
                feat_vec[2] = row_serProp[ei+1]
                feat_vec[3] = row_classcapa[ei+1]
                realnode_feat.append(feat_vec)
        
        len_eachdev = []
        devs = torch.tensor([], dtype = torch.int)
        paths_Tensor = torch.LongTensor(paths)
        len_eachpath_Tensor = torch.LongTensor(len_eachpath)
        len_eachpath_Tensor = len_eachpath_Tensor.cumsum(dim=0)
        for i in range(1,row_nn[0]+1):
            pos_dev = (paths_Tensor==i).nonzero()
            len_eachdev.append(len(pos_dev))
            max_num_services = 20
            devs_row = torch.zeros(max_num_services) 
            devs_row = devs_row.unsqueeze(0)
            for idx, (a, b) in enumerate(pos_dev):
                if a==0:
                    devs_row[0,idx] = (b+1)/2
                else:
                    devs_row[0,idx] = (len_eachpath_Tensor[a-1]/2)+(b+1)/2
            devs = torch.cat((devs, devs_row), dim=0)
        devs = devs.to(torch.int64)

        sample = {
            'num_devnodes': row_nn[0],
            'len_eachpath': len_eachpath,
            'edge': torch.LongTensor(edge).transpose(0,1),
            'fragTodev_edge': torch.LongTensor(fragTodev_edge).transpose(0,1),
            'paths': torch.LongTensor(paths),
            'row': torch.LongTensor(row),  
            'col': torch.LongTensor(col),  
            'path_ref': torch.LongTensor(path_ref),  
            'arr': torch.tensor(hypernode_feat) , 
            'node': torch.tensor(realnode_feat, dtype=torch.float),  
            'devs': devs,
            'len_eachdev': len_eachdev,
            'perform_respt': torch.tensor(perform_respt),
            'perform_tput': torch.tensor(perform_tput),  
            }
        data_list.append(sample)
    return data_list

def CollateBatch(batch):
    node = torch.tensor([], dtype=torch.float)

    len_eachpath = []
    len_eachdev = []
    devnode_idx = torch.tensor([], dtype = torch.int)
    fragnode_idx = torch.tensor([], dtype = torch.int)
    row = torch.tensor([], dtype = torch.int)
    col = torch.tensor([], dtype = torch.int)
    path_ref = torch.tensor([], dtype = torch.int)
    paths = torch.tensor([], dtype = torch.int)
    devs = torch.tensor([], dtype = torch.int)
    edge = torch.tensor([], dtype = torch.int)
    fragTodev_edge = torch.tensor([], dtype = torch.int)
    fragTodev_edge_idx = torch.tensor([], dtype = torch.int)

    total_nc = 0
    total_nn = 0
    total_ndevn = 0
    total_nfragn = 0

    arr = torch.tensor([], dtype = torch.float)
    perform_respt = torch.tensor([], dtype = torch.float)
    perform_tput = torch.tensor([], dtype = torch.float)

    for i in range(len(batch)):
        node = torch.cat((node, batch[i][0]['node']), dim=0)

        len_eachpath.extend(batch[i][0]['len_eachpath'])
        len_eachdev.extend(batch[i][0]['len_eachdev'])
        
        devnode_idx = torch.cat((devnode_idx, torch.tensor(np.arange(1,batch[i][0]['num_devnodes']+1)+total_nn)))
        fragnode_idx = torch.cat((fragnode_idx, torch.tensor(np.arange(batch[i][0]['num_devnodes']+1, len(batch[i][0]['node'])+1)+total_nn)))
        row = torch.cat((row, batch[i][0]['row']+total_nc))
        col = torch.cat((col, batch[i][0]['col']))
        path_ref = torch.cat((path_ref, batch[i][0]['path_ref']+total_nn))

        path = batch[i][0]['paths'].clone()
        dummy = torch.nonzero(path, as_tuple=True)
        r = dummy[0]
        c = dummy[1]
        path[r,c] = path[r,c]+total_nn
        paths = torch.cat((paths, path), dim=0)

        dev_copy = batch[i][0]['devs'].clone()
        dummy = torch.nonzero(dev_copy, as_tuple=True)
        r = dummy[0]
        c = dummy[1]
        dev_copy[r,c] = dev_copy[r,c]+total_nfragn
        devs = torch.cat((devs, dev_copy), dim=0)

        edge = torch.cat((edge, batch[i][0]['edge']+total_nn), dim=1)
        fragTodev_edge = torch.cat((fragTodev_edge, batch[i][0]['fragTodev_edge']+total_nn), dim=1)
        fragTodev_edge_idx = torch.cat((fragTodev_edge_idx, batch[i][0]['fragTodev_edge'][1]-1+total_ndevn))

        total_nc = total_nc+len(batch[i][0]['paths'])
        total_nn = total_nn+len(batch[i][0]['node'])
        total_ndevn = total_ndevn+batch[i][0]['num_devnodes']
        total_nfragn = total_nfragn+(len(batch[i][0]['node'])-batch[i][0]['num_devnodes'])

        arr = torch.cat((arr, batch[i][0]['arr']), dim=0)
        perform_respt = torch.cat((perform_respt, batch[i][1]))
        perform_tput = torch.cat((perform_tput, batch[i][2]))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    node = node.to(device)
    row = row.to(device)
    col = col.to(device)
    path_ref = path_ref.to(device)
    paths = paths.to(device)
    arr_elements = arr
    arr = torch.zeros(arr.shape[0], 5) 
    arr[:, 4] = arr_elements[:, 0]
    arr = arr.to(device)
    edge = edge.to(device)
    devnode_idx = devnode_idx.to(device)
    fragnode_idx = fragnode_idx.to(device)
    fragTodev_edge = fragTodev_edge.to(device)
    fragTodev_edge_idx = fragTodev_edge_idx.to(device)
    devs = devs.to(device)
    perform = torch.cat((perform_respt, perform_tput), dim=0)
    perform = perform.to(device)

    return node, row, col, path_ref, paths, arr, edge, devnode_idx, fragnode_idx, fragTodev_edge, fragTodev_edge_idx, len_eachpath, devs, len_eachdev, perform
