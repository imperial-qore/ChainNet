# ChainNet
ChainNet is a customized graph neural network model designed to evaluate the reliability of edge AI deployments. An edge AI service consists of a chain of deep neural network (DNN) fragments. For a given set of service chains, deployment refers to allocating their fragments across available devices. <br>

The input to ChainNet is a deployment plan characterized by a set of files covering two main aspects: (1) the mapping relationship between fragments and devices, and (2) the features of services, fragments, and devices. The output includes the throughput and latency of each service chain. <be>

# Quick start
Clone repo:
```bash
git clone https://github.com/imperial-qore/ChainNet.git
cd ChainNet/
```

Create an environment in line with the environment.yaml file:
```bash
conda env create -f environment.yaml
conda activate ChainNet_env
```

To train and test ChainNet, use the following scripts: 
```bash
python main.py
python evaluation.py
```

# Cite this work
Our work is accepted by the 54th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN). Cite our work using the bibtex entry below.
```bash
@inproceedings{chainnet2024dsn,
  title={ChainNet: A Customized Graph Neural Network Model for Loss-aware Edge AI Service Deployment},
  author={Niu, Zifeng and Roveri, Manuel and Casale, Giuliano},
  booktitle={IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)},
  year={2024},
  organization={IEEE}
}
```

# License
BSD-3-Clause. Copyright (c) 2024, Zifeng Niu. All rights reserved.<be>

See the license file for more details.
