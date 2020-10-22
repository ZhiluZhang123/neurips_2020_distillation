import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_acc(softmax, labels):
    _, predicted = torch.max(softmax, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def evaluate_test(model, testloader, temp=1):
    model.eval()
    correct = 0
    total = 0
    outputs_list = []
    labels_list = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = F.softmax(model(images) / temp, dim = 1).data

            outputs_list.append(outputs)
            labels_list.append(labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

    outputs = torch.cat(outputs_list).cuda()
    labels = torch.cat(labels_list).cuda()
    outputs_var = Variable(outputs)
    labels_var = Variable(labels)

    return outputs_var, labels_var