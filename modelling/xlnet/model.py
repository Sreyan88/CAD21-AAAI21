from config import *
from torch import nn
from transformers import *
import torch
import numpy as np
from torch import *
from torch.nn.utils.rnn import pack_padded_sequence
device = torch.device("cuda:1")



class SequenceWise(nn.Module): #Same as TimeDistributed
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module
    def forward(self, x):
        t, n = x.size(0), x.size(1)
        #print(t,n)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.reshape(t, n, -1)
        return x
    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr



class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return weighted_input




class transformer_model(nn.Module):
  def __init__(self, model_name, drop_prob = dropout_prob):
    super(transformer_model, self).__init__()

    configuration = XLNetConfig.from_pretrained(model_name, output_hidden_states=True)
    self.xlnet = XLNetModel.from_pretrained(model_name, config = configuration)

    # freezes layers of the model
    if to_freeze:
      cnt=0
      for child in xlnet.xlnet.children():
        cnt = cnt + 1
        if cnt<=freeze_layers:
          for param in child.parameters():
            param.requires_grad = False



    self.fc1 = nn.Linear(512, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc3 = nn.Linear(hidden_dim2, final_size)
    self.dropout = nn.Dropout(p=drop_prob)
    self.lstm = nn.LSTM(xlnet_dim, 256 , num_layers = 2, bidirectional=True, batch_first=True)
    self.lstm_attention = Attention(512,180)

    self.fullyconnnected = nn.Sequential(nn.Dropout(p=drop_prob),self.fc1,nn.ReLU(),nn.Dropout(p=drop_prob),self.fc2,nn.ReLU(),nn.Dropout(p=drop_prob),self.fc3)

    self.tmd = SequenceWise(self.fullyconnnected)

    #self.lstm_attention = Attention(512, 180)

  def avg(self, a, st, end):
    k = a
    lis = []
    for i in range(st,end):
      lis.append(a[i])
    x = torch.mean(torch.stack(lis),dim=0)
    return x

  def forward(self, xlnet_ids, xlnet_mask, xlnet_token_starts, lm_lengths = None, labels = None):

    batch_size = xlnet_ids.size()[0]
    pad_size = xlnet_ids.size()[1]
    # print("batch size",batch_size,"\t\tpad_size",pad_size)

    output = self.xlnet(xlnet_ids, attention_mask = xlnet_mask)

    # Concatenating hidden dimensions of all encoder layers
    xlnet_out = output[-1][0]
    for layers in range(1,13,1):
      xlnet_out = torch.cat((xlnet_out, output[-1][layers]), dim=2)

    #print(xlnet_out.shape)

    # Fully connected layers with relu and dropouts in between

    pred_logits,_ = self.lstm(self.dropout(xlnet_out))
    #pred_logits = self.lstm_attention(pred_logits)
    #print(pred_logits.shape)
    #pred_logits = torch.relu(pred_logits)
    pred_logits = self.tmd(pred_logits)
    #pred_logits = torch.relu(self.fc1(self.dropout(pred_logits)))
    #pred_logits = torch.relu(self.fc2(self.dropout(pred_logits)))
    pred_logits = torch.sigmoid(pred_logits)
    pred_logits = torch.squeeze(pred_logits,2)

    pred_labels = torch.tensor(np.zeros(xlnet_token_starts.size()),dtype = torch.float64).to(device)

    for b in range(batch_size):
      for w in range(pad_size):
        if(xlnet_token_starts[b][w]!=0):
          if(xlnet_token_starts[b][w]>=pad_size):
            #print(".")
            pass
          else:
            st = xlnet_token_starts[b][w]
            end = xlnet_token_starts[b][w+1]
            pred_labels[b][w] = pred_logits[b][xlnet_token_starts[b][w]]


    if(labels != None):
      lm_lengths, lm_sort_ind = lm_lengths.sort(dim=0, descending=True)
      scores = labels[lm_sort_ind]
      targets = pred_labels[lm_sort_ind]
      scores = pack_padded_sequence(scores, lm_lengths, batch_first=True).data
      targets = pack_padded_sequence(targets, lm_lengths, batch_first=True).data
      loss_fn = nn.BCELoss().to(device)
      loss = loss_fn(targets,scores)

      return loss, pred_labels

    else:
      return 0.0, pred_labels
