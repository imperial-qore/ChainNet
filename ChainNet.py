import torch
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
from inits import glorot, zeros

class Net(nn.Module):
    def __init__(self, num_iterations, size_realnode, size_hypernode, num_readoutneurons, num_heads, negative_slope, dropout):
        super().__init__()

        self.phiC = nn.GRU(size_realnode*2, size_hypernode, batch_first = True)
        self.phiF = nn.GRUCell(size_hypernode+size_realnode, size_realnode)
        self.phiD = nn.GRUCell(size_hypernode+size_realnode, size_realnode)

        self.N = num_iterations

        self.size_realnode = size_realnode
        self.size_hypernode = size_hypernode
        self.initmapping_device = nn.Linear(5, size_realnode)
        self.initmapping_fragment = nn.Linear(5, size_realnode)
        self.initmapping_service = nn.Linear(5, size_hypernode)

        self.num_heads = num_heads
        self.lin_source = Linear(size_hypernode+size_realnode, num_heads * (size_hypernode+size_realnode), bias=True, weight_initializer='glorot')
        self.lin_target = Linear(size_realnode, num_heads * (size_hypernode+size_realnode), bias=True, weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, num_heads, size_hypernode+size_realnode))
        self.bias = Parameter(torch.Tensor(size_hypernode+size_realnode))
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.throughput_mlp = nn.Sequential(
            nn.Linear(size_hypernode, num_readoutneurons),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(num_readoutneurons, num_readoutneurons),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(num_readoutneurons, 1)
        )

        self.latency_mlp = nn.Sequential(
            nn.Linear(size_realnode, num_readoutneurons),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(num_readoutneurons, num_readoutneurons),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(num_readoutneurons, 1)
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_source.reset_parameters()
        self.lin_target.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        realnode = x[0]
        index_devices = x[7]
        index_fragments = x[8]
        devices = torch.index_select(realnode, dim=0, index=index_devices-1)
        fragments = torch.index_select(realnode, dim=0, index=index_fragments-1)
        devices = self.initmapping_device(devices)
        fragments = self.initmapping_fragment(fragments)
        select = torch.cat([devices,fragments],dim=0)
        select_index = torch.cat((index_devices,index_fragments))
        realnode_state = torch_scatter.scatter(src=select, index=select_index, dim=0, reduce='sum')

        len_exeSeq = x[11]
        len_exeSeq_list = [int(element/2) for element in len_exeSeq]
        max_len_exeSeq = max(len_exeSeq_list)
        arr = x[5]
        service_state = self.initmapping_service(arr).unsqueeze(0)
        exeSeqs = x[4]
        fragTodev_edges = x[9]
        fragTodev_device_index = x[10]

        for _ in range(self.N):
            exeSeqs_matrix = realnode_state[exeSeqs]  
            exeSeqs_matrix_dim1 = exeSeqs_matrix.size(0)
            exeSeqs_matrix_dim2 = exeSeqs_matrix.size(1)
            exeSeqs_matrix_dim3 = exeSeqs_matrix.size(2)
            exeSeqs_matrix = exeSeqs_matrix.view(exeSeqs_matrix_dim1,int(exeSeqs_matrix_dim2/2),exeSeqs_matrix_dim3*2)
            # message-passing for service chains
            packed = pack_padded_sequence(exeSeqs_matrix, len_exeSeq_list, batch_first=True, enforce_sorted=False)
            seq_packed, service_state = self.phiC(packed, service_state) 

            # reshape output
            output, lens_unpacked = pad_packed_sequence(seq_packed, batch_first=True)
            output = output.reshape(-1,self.size_hypernode)
            
            # pick meaningful output
            extract_indices = []
            for extract in range(len(len_exeSeq_list)): 
                extract_indices.extend(extract_id for extract_id in range(extract*max_len_exeSeq,extract*max_len_exeSeq+len_exeSeq_list[extract]))
            output = output[extract_indices] 

            # message-passing for fragments
            ## messageF_fromservice
            messageF_fromservice = output
            ## messageF_fromdevice
            source = torch.index_select(realnode_state, dim=0, index=fragTodev_edges[1])
            cap_toedge_index = torch.tensor(np.arange(0,fragTodev_edges.size(1)),dtype=torch.int64).to(device)
            messageF_fromdevice = torch_scatter.scatter(src=source, index=cap_toedge_index, dim=0, reduce='sum') 
            ## concatenation
            messageF = torch.cat((messageF_fromservice, messageF_fromdevice),dim=1)
            ## update
            fragment_state = torch.index_select(realnode_state, dim=0, index=index_fragments)
            fragment_state_copy = fragment_state
            fragment_state = self.phiF(messageF, fragment_state) 

            # message-passing for devices
            ## concatenation
            messageD_exeSteps = torch.cat((output,fragment_state_copy),dim=1)
            ## linear transformation
            messageD_exeSteps_trans = self.lin_source(messageD_exeSteps).view(-1, self.num_heads, self.size_hypernode+self.size_realnode) # x_j (j->i)
            device_state = torch.index_select(realnode_state, dim=0, index=index_devices)
            device_state_trans = self.lin_target(device_state).view(-1, self.num_heads, self.size_hypernode+self.size_realnode)
            ## attention coefficients of (src,dst) pairs
            device_state_trans_select = torch.index_select(device_state_trans, dim=0, index=fragTodev_device_index) 
            x = messageD_exeSteps_trans+device_state_trans_select
            x = F.leaky_relu(x, self.negative_slope)
            alpha = (x * self.att).sum(dim=-1)
            ## transform attention coefficients to weights (softmax)
            alpha_max = torch_scatter.scatter(alpha, index=fragTodev_device_index, dim=0, reduce='max')
            alpha_max = torch.index_select(alpha_max, dim=0, index=fragTodev_device_index)
            out = (alpha - alpha_max).exp()
            out_sum = torch_scatter.scatter(out, index=fragTodev_device_index, dim=0, reduce='sum')
            out_sum = torch.index_select(out_sum, dim=0, index=fragTodev_device_index)
            alpha = out / (out_sum + 1e-16)
            ## weighted
            weighted_messageD_exeSteps_trans = messageD_exeSteps_trans * alpha.unsqueeze(-1) 
            ## weighted sum
            messageD = torch_scatter.scatter(src=weighted_messageD_exeSteps_trans, index=fragTodev_device_index, dim=0, reduce='sum') # aggregate sernode, ->capnode
            messageD = messageD.mean(dim=1) # for multi-head cases 
            messageD = messageD+self.bias
            ## update
            device_state = self.phiD(messageD, device_state) 

            # renew the embeddings of real nodes
            select = torch.cat([device_state,fragment_state],dim=0)
            select_index = torch.cat((index_devices,index_fragments))
            realnode_state = torch_scatter.scatter(src=select, index=select_index, dim=0, reduce='sum')
        
        start_index = 0
        pooled_result = []
        for group in len_exeSeq_list:
            pooled_tensor = fragment_state[start_index:start_index+group].mean(dim=0)
            pooled_result.append(pooled_tensor)
            start_index = start_index + group
        latency_state = torch.stack(pooled_result)
        latency = self.latency_mlp(latency_state).squeeze(-1)
        latency = latency.unsqueeze(0)

        throughput = self.throughput_mlp(service_state).squeeze(-1)

        solution = torch.cat((latency, throughput), dim=1)

        return solution
    