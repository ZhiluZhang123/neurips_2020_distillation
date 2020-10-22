import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import argparse

from models import *
from dataloader import *
from loss import *
from ece import *
from tools import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset used')
parser.add_argument('--loss_type', default='beta', type=str)
parser.add_argument('--alpha', default=0.6, type=float)
parser.add_argument('--beta', default=3.0, type=float)
parser.add_argument('--lamda', default=0.999, type=float)
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--num_epochs', default=150, type=int, help='num_epochs')
parser.add_argument('--num_repeat', type=int, default=5)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0001, type=float, help='learning rate')
parser.add_argument('--valid_size', default=5000, type=int, help='valid_size')
parser.add_argument('--dont_save', action="store_true", help='dont save model')

args = parser.parse_args()

batch_size = args.batch_size
valid_size = args.valid_size
num_epochs = args.num_epochs
initial_rate = args.lr
loss_type = args.loss_type
wd = args.wd
beta = args.beta
alpha = args.alpha

path_name = args.model + '_lossType' + loss_type + '_dataset' + args.dataset + '_beta' + \
            str(args.beta) + '_alpha' + str(args.alpha)

### load data ###
trainloader, valloader, testloader, num_classes = load_data(args.dataset, valid_size, batch_size) 

### define metrics ###
criterion = nn.CrossEntropyLoss()
nll_criterion = nn.NLLLoss().cuda()
ece_criterion = ECELoss().cuda()

### train model ###
for repeat in range(args.num_repeat):
    file_name = path_name + '_expNum' + str(repeat)
    print(file_name)
    
    trn_loss = []
    val_loss = []
    val_acc_list = []
    
    # initialize model 
    if args.model == 'resnet':
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
        model.cuda() 
        optimizer = optim.SGD(model.parameters(), lr=initial_rate, momentum=0.9, weight_decay=wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs/3, num_epochs/3*2], gamma=0.1)
        
        if args.loss_type == 'beta':
            ema_model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
            ema_model.cuda()   
        
    elif args.model == 'densenet':
        model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=num_classes)
        model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=initial_rate, momentum=0.9, weight_decay=wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs*0.5, num_epochs*0.75], gamma=0.1)
        
        if args.loss_type == 'beta':
            ema_model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=num_classes)
            ema_model.cuda()
            
    if args.loss_type == 'beta':
        for param in ema_model.parameters():
            param.detach_()
    
    global_step = 0
    for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
        scheduler.step()
        running_loss = 0.0
        correct = 0
        total = 0
        correct1 = 0
        for i, data in enumerate(trainloader, 0):
            
            model.train()
            ema_model.train()
            
            # get the inputs
            inputs, labels = data
                
            # wrap them in Variable 
            inputs_var, labels_var = inputs.cuda(), labels.cuda()          
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs_var)
            
            if args.loss_type == 'beta':
                with torch.no_grad():
                    outputs_orig = ema_model(inputs_var)
            
            if loss_type == 'beta':
                beta_loss = soft_beta_loss(outputs, labels_var, beta, outputs_orig, num_classes=num_classes)
                loss = (1-alpha)*criterion(outputs, labels_var) + alpha*beta_loss
                
            elif loss_type == 'smoothing':
                loss = soft_cce_loss(outputs, labels_var, num_classes=num_classes, epsilon=beta)
                
            elif loss_type == 'ent_reg':
                loss = criterion(outputs, labels_var) - beta*get_entropy(outputs)

            else:
                print('loss not implemented.')
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data.cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            
            # update ema_model model
            if args.loss_type == 'beta':
                _, predicted = torch.max(outputs_orig.data.cpu(), 1)
                correct1 += (predicted == labels).sum()
                
                global_step += 1
                alpha_now = min(1 - 1 / (global_step + 1), args.lamda)
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(alpha_now).add_(1 - alpha_now, param.data)
            
        trn_loss += [running_loss/(i+1)]
        
        #Validation
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(valloader, 0):     

            model.eval()
            inputs, labels = data

            inputs_var, labels_var = Variable(inputs.cuda()), Variable(labels.cuda())

            outputs = model(inputs_var)

            loss = criterion(outputs, labels_var)

            _, predicted = torch.max(outputs.data.cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            # Calculate statistics
            running_loss += loss.item()

        val_loss += [running_loss/(i+1)]
        val_acc_list += [float(correct) / total]

        # save model based on validation accuracy
        if len(val_loss) == 1:
            #print('saved')
            torch.save(model.state_dict(), './saved_models/'+ file_name + '.pth.tar')

        else:
            if val_acc_list[-1] >= max(val_acc_list):
                #print('updated')
                torch.save(model.state_dict(), './saved_models/'+ file_name + '.pth.tar')
        
    ### evaluate model ###
    model.load_state_dict(torch.load('./saved_models/'+ file_name + '.pth.tar'))
    model.eval()

    softmaxes, labels = evaluate_test(model, testloader)
    acc = get_acc(softmaxes, labels)
    nll = nll_criterion(torch.log(softmaxes), labels).item()
    ece = ece_criterion(softmaxes, labels).item()
    print('Accuracy on testset is: %.3f, NLL: %.3f, ECE: %.3f' % (acc, nll, ece))
    print('--------------------------------------------')