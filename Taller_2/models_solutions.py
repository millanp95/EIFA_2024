import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

sys.path.append('src/')
sys.path.append('../src/')

from LossFunctions import IID_loss, info_nce_loss
from torch.utils.data import DataLoader



from utils import SequenceDataset, create_dataloader

# Random Seeds for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

global dtype
global EPS

dtype = torch.FloatTensor
EPS = sys.float_info.epsilon

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.cuda.FloatTensor



def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class NetLinear(nn.Module):
    def __init__(self, n_input, n_output):
        super(NetLinear, self).__init__()
        self.n_input = n_input
        self.layers  = nn.Sequential(

            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64)            
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, n_output),  
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, self.n_input)
        latent = self.layers(x)
        out = self.classifier(latent)
        return out, latent
    

class myNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(myNet, self).__init__()
        self.n_input = n_input
        self.layers = nn.Sequential(
                nn.Linear(n_input, 400),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(400, 128),
                nn.LeakyReLU()
        )
        
        self.instance = nn.Linear(128, 64) 
        
        self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128, n_output),  
                nn.Softmax(dim=1)
        )
            
    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = self.layers(x)
        latent = self.instance(x)
        out = self.classifier(x)
        return out, latent
    

class IID_model():
    def __init__(self, args: dict):

        self.sequence_file = args['sequence_file']
        self.GT_file = args['GT_file']

        self.n_clusters = args['n_clusters']
        self.k = args['k']
        
        self.n_features = 4**self.k
        self.net = NetLinear(self.n_features, args['n_clusters'])
        self.reduce = False
            
        
        self.net.apply(weights_init)
        self.net.to(device)
        self.epoch = 0
        self.EPS = sys.float_info.epsilon
        
        self.n_mimics = args['n_mimics']
        self.batch_sz = args['batch_sz']
        self.optimizer = 'SGD'
        self.l = 2.8 
        self.lr = 1e-3
        self.weight = 0.5
        self.schedule = None
        self.mutate = True

        if self.optimizer == 'RMSprop': 
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=0.01, momentum=0.9)
        elif self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer not supported")
        
        if self.schedule == 'Plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        elif self.schedule == 'Triangle':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5, mode="triangular2")
        
    def build_dataloader(self):
        #Data Files
        data_path = self.sequence_file
        GT_file = self.GT_file
                
        self.dataloader = create_dataloader(data_path, 
                                             self.n_mimics, 
                                             k=self.k, 
                                             batch_size=self.batch_sz, 
                                             GT_file=GT_file,
                                             reduce=self.reduce)

    def contrastive_training_epoch(self):
        n_features = self.n_features
        batch_size = self.batch_sz
        k = self.k 
        self.net.train()
        running_loss = 0.0

        for i_batch, sample_batched in enumerate(self.dataloader):
            sample = sample_batched['true'].view(-1, 1, self.n_features).type(dtype)
            modified_sample = sample_batched['modified'].view(-1, 1, self.n_features).type(dtype)
            
            # zero the gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            z1, h1 = self.net(sample)
            z2, h2 = self.net(modified_sample)

            loss = (1-self.weight)*info_nce_loss(h1, h2, 0.85) + (self.weight)*IID_loss(z1, z2, lamb=self.l)
            loss.backward()
            self.optimizer.step()

            running_loss += loss
            
        running_loss /= i_batch

        if self.schedule == 'Plateau':
            self.scheduler.step(running_loss)
        elif self.schedule == 'Triangle':
            self.scheduler.step()
        self.epoch += 1

        return running_loss.item()

    def predict(self, data=None):
        
        n_features = self.n_features
        test_dataset = SequenceDataset(self.sequence_file, k=self.k, transform=None, GT_file=self.GT_file, reduce=self.reduce)
        test_dataloader = DataLoader(test_dataset, 
                             batch_size=self.batch_sz,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)
        y_pred = []
        probabilities = []
        latent = []

        with torch.no_grad():
            self.net.eval()
            
            for test in test_dataloader:
                kmers = test['kmer'].view(-1, 1, self.n_features).type(dtype)
                outputs, logits = self.net(kmers)
                probs,  predicted = torch.max(outputs, 1)

                #Extend our list with predictions and groud truth
                y_pred.extend(predicted.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())
                latent.extend(logits.cpu().tolist())
                
        return np.array(y_pred), np.array(probabilities), np.array(latent) 

    def calculate_probs(self, data=None):
        
        n_features = self.n_features
        test_dataset = SequenceDataset(self.sequence_file, k=self.k, transform=None, GT_file=self.GT_file, reduce=self.reduce)
        test_dataloader = DataLoader(test_dataset, 
                             batch_size=self.batch_sz,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)
 
        probabilities = []
        with torch.no_grad():
            self.net.eval()
            for test in test_dataloader:
                kmers = test['kmer'].view(-1, 1, n_features).type(dtype)

                #calculate the prediction by running through the network
                outputs, logits = self.net(kmers)
                probabilities.extend(outputs.cpu().tolist())

        return np.array(probabilities)