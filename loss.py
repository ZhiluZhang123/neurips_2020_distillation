import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def soft_beta_loss(outputs, labels, beta, outputs_orig, num_classes=10):

    softmaxes = F.softmax(outputs, dim=1)
    n, num_classes = softmaxes.shape
    tensor_labels = Variable(torch.zeros(n, num_classes).cuda().scatter_(1, labels.long().view(-1, 1).data, 1))

    # sort outputs and labels based on confidence/entropy        
    softmaxes_orig = F.softmax(outputs_orig, dim=1)
    maximum, _ = (softmaxes_orig*tensor_labels).max(dim=1)
    maxes, indices = maximum.sort()

    sorted_softmax, sorted_labels = softmaxes[indices], tensor_labels[indices]
    sorted_softmax_orig = softmaxes_orig[indices]
    
    # generate beta labels  
    random_beta = np.random.beta(beta, 1, n)
    random_beta.sort()
    random_beta = torch.from_numpy(random_beta).cuda()
    
    # create beta smoothing labels 
    uniform = (1 - random_beta) / (num_classes - 1)
    random_beta -= uniform
    random_beta = random_beta.view(-1, 1).repeat(1, num_classes).float()
    beta_label = sorted_labels*random_beta
    beta_label += uniform.view(-1, 1).repeat(1, num_classes).float()
    
    # compute NLL loss
    loss = -beta_label * torch.log(sorted_softmax + 10**(-8))
    loss = loss.sum() / n

    return loss

def get_entropy(outputs):
    
    loss = F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)
    loss = -loss.sum() / loss.shape[0]

    return loss

def soft_cce_loss(outputs, labels, num_classes = 10, epsilon=0.0):
    outputs = F.softmax(outputs, dim=1)

    if labels[0].dim() == 0:
        labels = Variable(torch.zeros(labels.size(0), num_classes).cuda().scatter_(1, labels.long().view(-1, 1).data, 1))
       
    labels = labels*(1- epsilon - epsilon/(num_classes - 1))
    labels = labels + epsilon/(num_classes - 1)
    
    loss = -labels * torch.log(outputs + 10**(-8))

    loss = loss.sum() / loss.shape[0]

    return loss