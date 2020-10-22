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

parser.add_argument('--dataset', default='cifar100', type=str, help='dataset used')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--num_epochs', default=150, type=int, help='num_epochs')
parser.add_argument('--temp', default=4.0, type=float)
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--valid_size', default=5000, type=int, help='valid_size')
parser.add_argument('--wd', default=0.0001, type=float, help='learning rate')
parser.add_argument('--num_repeat', type=int, default=5)
parser.add_argument('--num_gen', type=int, default=2)

args = parser.parse_args()

batch_size = args.batch_size
valid_size = args.valid_size
num_epochs = args.num_epochs
initial_rate = args.lr
wd = args.wd

path_name = 'distillation' + args.model + '_' + args.dataset +'_numEpochs' + '_temp' + str(args.temp) 

### load data ###
trainloader, valloader, testloader = load_data(args.dataset, valid_size) 
    
### define metrics ###
criterion = nn.CrossEntropyLoss()
nll_criterion = nn.NLLLoss().cuda()
ece_criterion = ECELoss().cuda()

for repeat in range(args.num_repeat):
    file_name = path_name + '_expNum' + str(repeat)
    print(file_name)
    
    for gen in range(num_gen):
        trn_loss = []
        val_loss = []
        val_acc_list = []
        
        ### initialize model ###
        if args.model == 'resnet':
            model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
            model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=initial_rate, momentum=0.9, weight_decay=wd)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs/3, num_epochs/3*2], gamma=0.1)

        elif args.model == 'densenet':   
            model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=num_classes)
            model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=initial_rate, momentum=0.9, weight_decay=wd)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs*0.5, num_epochs*0.75], gamma=0.1)

        ### load teacher model ###
        if gen > 0:
            if args.model == 'resnet':
                teacher = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
                teacher.cuda()
               
            elif args.model == 'densenet':
                teacher = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=num_classes)
                teacher.cuda()

            teacher_path = './saved_models/'+ file_name + 'gen_' + str(gen - 1) + '.pth.tar'
            teacher.load_state_dict(torch.load(teacher_path))
            teacher.eval()
        
        ### train model ###
        for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
            scheduler.step()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, data in enumerate(trainloader, 0):
                model.train()
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs_var, labels_var = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs_var)
                if gen == 0:
                    loss = criterion(outputs, labels_var) 

                else:
                    with torch.no_grad():
                        outputs_teacher = teacher(inputs_var)
                        
                    cce_loss = criterion(outputs, labels_var)
                    soft_labels = F.softmax(outputs_teacher / temp, dim = 1)
                    soft_loss = soft_cce_loss(outputs, soft_labels)
                    loss = (1- args.beta)*cce_loss + args.beta*soft_loss
                     
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data.cpu(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

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
            val_acc_list += [float(correct)/total]
            
            # save model
            if not args.dont_save:
                if len(val_loss) == 1:
                    # print('saved')
                    torch.save(model.state_dict(), './saved_models/'+ file_name + 'gen_' + str(gen) + '.pth.tar')

                else:
                    if val_acc_list[-1] >= max(val_acc_list):
                        #print('updated')
                        torch.save(model.state_dict(), './saved_models/'+ file_name + 'gen_' + str(gen) + '.pth.tar')

        
        ### evaluate trained model on test set ###
        student_path = './saved_models/'+ file_name + 'gen_' + str(gen) + '.pth.tar'
        model.load_state_dict(torch.load(student_path))
        model.eval()

        softmaxes, labels = evaluate_test(model, testloader)
        acc = get_acc(softmaxes, labels)
        nll = nll_criterion(torch.log(softmaxes), labels).item()
        ece = ece_criterion(softmaxes, labels).item()
        print('Student with temp = %.3f: acc: %.3f, NLL: %.3f, ECE: %.3f' % (temp, acc, nll, ece))
        print('--------------------------------------------')






