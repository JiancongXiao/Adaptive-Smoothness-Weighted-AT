import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import ipdb
import random
import sys 
sys.path.append('./utils/')
from swa_utils import *

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]    

'''
#DEFAULTS
pgd_linf: epsilon=0.03, alpha=0.003, num_iter = 40
pgd_l0  : epsilon = 12, alpha = 1
pgd_l1_topk  : epsilon = 12, alpha = 0.05, alpha_ = 1, num_iter = 50, k = rand(5,20) --> (alpha = alpha_ /k *20)
pgd_l2  : epsilon =0.5, alpha=0.05, num_iter = 50

'''

def label_noise(y, gamma):
    gamma = (10*gamma-1)/9
    bs = y.size()
    bs = bs[0]
    rnd = torch.rand(bs).cuda()
    temp = 10*(rnd[rnd>gamma]-gamma)/(1-gamma)
    y[rnd>gamma] = temp.long()
    y[y==10] = 9
    return y

class LabelSmoothing(nn.Module):
    def __init__(self,smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs =  torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def pgd_l2(model, X, y, epsilon=0.5, alpha=0.05, num_iter = 50, device = "cuda:0", restarts = 0, version = 0):
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1 (Test time)
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data *=  epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.data =   torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]     
        delta.grad.zero_()  

    max_delta = delta.detach()

    for k in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)*epsilon 
        delta.data /= norms(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    return max_delta    


def pgd_l1_topk(model, X,y, epsilon = 12, alpha = 0.05, num_iter = 50, k = 20, device = "cuda:1", restarts = 1, version = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    gap = alpha
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)
    alpha_default = alpha
    for t in range (num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        k = random.randint(5,20)
        alpha = alpha_default/k*20
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_() 

    max_delta = delta.detach()

    #Restarts    
    for k in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)*epsilon 
        delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)
        for t in range (num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            k = random.randint(5,20)
            alpha = alpha_default/k*20
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta

def pgd_linf(model, X, y, epsilon=0.03, alpha=0.003, num_iter = 40, device = "cuda:0", restarts = 0, version = 0):
    """ Construct FGSM adversarial examples on the examples X"""
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    max_delta = delta.detach()
    
    for i in range (restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data*2.0  - 1.0)*epsilon
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only themonly for version 1
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,255]
            delta.grad.zero_()

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta    


def pgd_l0(model, X,y, epsilon = 12, alpha = 1, num_iter = 0, device = "cuda:1"):
    delta = torch.zeros_like(X, requires_grad = True)
    batch_size = X.shape[0]
    # print("Updated")
    for t in range (epsilon):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        temp = delta.grad.view(batch_size, 1, -1)
        neg = (delta.data != 0)
        X_curr = X + delta
        neg1 = (delta.grad < 0)*(X_curr < 0.1)
        neg2 = (delta.grad > 0)*(X_curr > 0.9)
        neg += neg1 + neg2
        u = neg.view(batch_size,1,-1)
        temp[u] = 0
        my_delta = torch.zeros_like(X).view(batch_size, 1, -1)
        
        maxv =  temp.max(dim = 2)
        minv =  temp.min(dim = 2)
        val_max = maxv[0].view(batch_size)
        val_min = minv[0].view(batch_size)
        pos_max = maxv[1].view(batch_size)
        pos_min = minv[1].view(batch_size)
        select_max = (val_max.abs()>=val_min.abs()).half()
        select_min = (val_max.abs()<val_min.abs()).half()
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = (1-X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max])*select_max
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min]*select_min
        delta.data += my_delta.view(batch_size, 3, 32, 32)
        delta.grad.zero_()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
    
    return delta.detach()

def msd_v0(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.5, epsilon_l_1 = 12, 
                alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 0.05, num_iter = 50, device = "cuda:0"):
    delta = torch.zeros_like(X,requires_grad = True)
    max_delta = torch.zeros_like(X)
    max_max_delta = torch.zeros_like(X)
    max_loss = torch.zeros(y.shape[0]).to(y.device).half()
    max_max_loss = torch.zeros(y.shape[0]).to(y.device).half()
    alpha_l_1_default = alpha_l_1
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():                
            #For L_2
            delta_l_2  = delta.data + alpha_l_2*delta.grad / norms(delta.grad)      
            delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)
            delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]

            #For L_inf
            delta_l_inf=  (delta.data + alpha_l_inf*delta.grad.sign()).clamp(-epsilon_l_inf,epsilon_l_inf)
            delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]

            #For L1
            k = random.randint(5,20)
            alpha_l_1 = (alpha_l_1_default/k)*20
            delta_l_1  = delta.data + alpha_l_1*l1_dir_topk(delta.grad, delta.data, X, alpha_l_1, k = k)
            delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
            delta_l_1  = torch.min(torch.max(delta_l_1, -X), 1-X) # clip X+delta to [0,1]
            
            #Compare
            delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
            max_loss = torch.zeros(y.shape[0]).to(y.device).half()
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp)
            delta.data = max_delta.data
            max_max_delta[max_loss> max_max_loss] = max_delta[max_loss> max_max_loss]
            max_max_loss[max_loss> max_max_loss] = max_loss[max_loss> max_max_loss]
        delta.grad.zero_()

    return max_max_delta



def mix(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.5, epsilon_l_1 = 12, 
                alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 1, num_iter = 50, device = "cuda:0"):
    pass

def sat(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.5, epsilon_l_1 = 12, 
                alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 1, num_iter = 50, device = "cuda:0"):
    pass

def adt(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.5, epsilon_l_1 = 12, 
                alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 1, num_iter = 50, device = "cuda:0"):
    pass

def pgd_worst_dir(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.5, epsilon_l_1 = 12, 
    alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 0.05, num_iter = 50, device = "cuda:0"):
    #Always call version = 0
    delta_1 = pgd_l1_topk(model, X, y, epsilon = epsilon_l_1, alpha = alpha_l_1,  device = device)
    delta_2 = pgd_l2(model, X, y, epsilon = epsilon_l_2, alpha = alpha_l_2,  device = device)
    delta_inf = pgd_linf(model, X, y, epsilon = epsilon_l_inf, alpha = alpha_l_inf, device = device)
    
    batch_size = X.shape[0]

    loss_1 = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_1), y)
    loss_2 = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_2), y)
    loss_inf = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_inf), y)

    delta_1 = delta_1.view(batch_size,1,-1)
    delta_2 = delta_2.view(batch_size,1,-1)
    delta_inf = delta_inf.view(batch_size,1,-1)

    tensor_list = [loss_1, loss_2, loss_inf]
    delta_list = [delta_1, delta_2, delta_inf]
    loss_arr = torch.stack(tuple(tensor_list))
    delta_arr = torch.stack(tuple(delta_list))
    max_loss = loss_arr.max(dim = 0)
    

    delta = delta_arr[max_loss[1], torch.arange(batch_size), 0]
    delta = delta.view(batch_size,3, X.shape[2], X.shape[3])
    return delta


def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 20) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval.half()).half() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)

def proj_l1ball(x, epsilon=10, device = "cuda:1"):
    assert epsilon > 0
#     ipdb.set_trace()
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
#         y = x* epsilon/norms_l1(x)
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y = y.view(-1,3,32,32)
    y *= x.sign()
    return y


def proj_simplex(v, s=1, device = "cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).half().to(device)
    comp = (vec > (cssv - s)).half()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.HalfTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.half() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w

def epoch(args, loader, lr_schedule,  model, epoch_i, criterion = nn.CrossEntropyLoss(), opt=None, device = "cuda:0", stop = False, log = None):
    """Standard training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        output = model(X)
        loss = criterion(output, y)        
        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if opt != None: 
            if (i + 1) % args.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}]  " \
                        "Loss:[{:.3f}]" \
                        "Acc:[{:.3f}]".format(
                            epoch_i, i + 1, len(loader), loss.item(), train_acc/train_n)

                log(p_str.format())

        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n


def epoch_adversarial_saver(args, batch_size,loader, model, attack, epsilon, num_iter, device = "cuda:0", restarts = 10, log = None):
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = 0
    train_n = 0
    # print("Attack: ", attack, " epsilon: ", epsilon )
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        delta = attack(model, X, y, epsilon = epsilon, num_iter = num_iter, device = device, restarts = restarts)
        output = model(X+delta)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        correct = (output.max(1)[1] == y).float()
        eps = (correct*1000 + epsilon - 0.000001).float()
        train_n += y.size(0)
        break
    return eps,  train_acc / train_n

def epoch_adversarial(args, loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
    opt=None, device = "cuda:0", stop = False, stats = False, log = None, label_noises = 'False', **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
#     ipdb.set_trace()
    
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']

        if stats:
            delta = attack(model, X, y, device = device, batchid = i, **kwargs)
        else:
            delta = attack(model, X, y, device = device, **kwargs)
        output = model(X+delta)
        # imshow(X[11])
        # print (X[11])
        # imshow((X+delta)[11])
        # print (norms_l1(delta))
#         output = model(X)
        if  label_noises == 'True':
            y = label_noise(y, gamma = 0.9)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            if (stop):
                break
        if opt != None:
            if (i + 1) % args.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}]  " \
                        "Loss:[{:.3f}]" \
                        "Acc:[{:.3f}]".format(
                            epoch_i, i + 1, len(loader), loss.item(), train_acc/train_n)

                log(p_str.format())

        
    return train_loss / train_n, train_acc / train_n
    



def epoch_adversarial_mix(args, loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
    opt=None, device = "cuda:0", stop = False, stats = False, log = None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
#     ipdb.set_trace()
    
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        if stats:
            delta = attack(model, X, y, device = device, batchid = i, **kwargs)
        else:
            delta1 = pgd_l1_topk(model, X, y, device = device)
            delta2 = pgd_l2(model, X, y, epsilon=0.5, device = device)
            delta3 = pgd_linf(model, X, y, device = device)
            delta = (delta1+delta2+delta3)/3
            delta = torch.min(torch.max(delta, -X), 1-X)
            # XX = torch.clamp(X+delta, 0, 1)
        output = model(X + delta)
        # imshow(X[11])
        # print (X[11])
        # imshow((X+delta)[11])
        # print (norms_l1(delta))
#         output = model(X)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            if (stop):
                break
        if opt != None:
            if (i + 1) % args.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}]  " \
                        "Loss:[{:.3f}]" \
                        "Acc:[{:.3f}]".format(
                            epoch_i, i + 1, len(loader), loss.item(), train_acc/train_n)

                log(p_str.format())
            
    return train_loss / train_n, train_acc / train_n



def epoch_adversarial_sat(args, loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
    opt=None, device = "cuda:0", stop = False, stats = False, log = None, label_noises = 'False', **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
#     ipdb.set_trace()
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        if stats:
            delta = attack(model, X, y, device = device, batchid = i, **kwargs)
        else:
            seed = torch.rand(1)
            if seed < 1/3:
                delta = pgd_l1_topk(model, X, y, device = device)
            elif seed < 2/3:
                delta = pgd_l2(model, X, y, epsilon=0.5, device = device)
            else: 
                delta = pgd_linf(model, X, y, device = device)

        output = model(X+delta)
        if  label_noises == 'True':
            y = label_noise(y, gamma = 0.9)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            if (stop):
                break
        if opt != None: 
            if (i + 1) % args.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}]  " \
                        "Loss:[{:.3f}]" \
                        "Acc:[{:.3f}]".format(
                            epoch_i, i + 1, len(loader), loss.item(), train_acc/train_n)

                log(p_str.format())

    return train_loss / train_n, train_acc / train_n




def triple_adv(args, loader, lr_schedule, model, epoch_i, attack,  criterion = nn.CrossEntropyLoss(),
                     opt=None, device= "cuda:0", stop = False, epsilon_l_1 = 12, epsilon_l_2 = 0.5, epsilon_l_inf = 0.03, num_iter = 50, log = None, label_noises = 'False'):
    
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        lr = lr_schedule(epoch_i + (i+1)/len(loader))
        if opt:
            opt.param_groups[0].update(lr=lr)
        ##Always calls the default version 0 for the individual attacks

        #L1
        delta = pgd_l1_topk(model, X, y, device = device, epsilon = epsilon_l_1)
        output = model(X+delta)
        if  label_noises == 'True':
            yn = label_noise(y, gamma = 0.9)
        else:
            yn = y
        loss_1 = criterion(output,yn)
        train_loss += loss_1.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        
        if opt:
            opt.zero_grad()
            loss_1.backward()
            opt.step()
        
        #L2
        delta = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2)
        output = model(X+delta)
        loss_2 = criterion(output,yn)
        train_loss += loss_2.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if opt:
            opt.zero_grad()
            loss_2.backward()
            opt.step()
        

        #Linf
        delta = pgd_linf(model, X, y, device = device, epsilon = epsilon_l_inf)
        output = model(X+delta)
        loss_inf = criterion(output,yn)
        train_loss += loss_inf.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if opt:
            opt.zero_grad()
            loss_inf.backward()
            opt.step()
        
        else:
            break
        if opt:
            if (i + 1) % args.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}]  " \
                        "Loss_1:[{:.3f}]" "Loss_2:[{:.3f}]" "Loss_inf:[{:.3f}]"\
                        "Acc:[{:.3f}]".format(
                            epoch_i, i + 1, len(loader), loss_1.item(), loss_2.item(), loss_inf.item(), train_acc/train_n)

                log(p_str.format())

    return train_loss / train_n, train_acc / train_n, loss_1.item(), loss_2.item(), loss_inf.item()

def epoch_adversarial_adt(args, loader, lr_schedule, model, epoch_i, attack,  criterion = nn.CrossEntropyLoss(),
                     opt=None, device= "cuda:0", stop = False, epsilon_l_1 = 12, epsilon_l_2 = 0.5, epsilon_l_inf = 0.03, num_iter = 50, log = None):
    
    train_loss = 0
    train_acc = 0
    train_n = 0
    w1 = 1/3
    w2 = 1/3
    winf = 1/3
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        lr = lr_schedule(epoch_i + (i+1)/len(loader))
        if opt:
            opt.param_groups[0].update(lr=lr)
        ##Always calls the default version 0 for the individual attacks

        #L1
        delta = pgd_l1_topk(model, X, y, device = device, epsilon = epsilon_l_1)
        output = model(X+delta)
        loss_1 = criterion(output,y)
        train_loss += loss_1.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        
        #L2
        delta = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2)
        output = model(X+delta)
        loss_2 = nn.CrossEntropyLoss()(output,y)
        train_loss += loss_2.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        #Linf
        delta = pgd_linf(model, X, y, device = device, epsilon = epsilon_l_inf)
        output = model(X+delta)
        loss_inf = nn.CrossEntropyLoss()(output,y)
        train_loss += loss_inf.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        loss = w1 * loss_1 + w2 * loss_2 + winf*loss_inf
        
        b1 = 1/loss_1.item()
        b2 = 1/loss_2.item() 
        binf = 1/loss_inf.item()
        sumloss = b1 + b2 + binf

        w1= b1/sumloss
        w2= b2/sumloss
        w1= binf/sumloss

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        else:
            break
        if opt:
            if (i + 1) % args.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}]  " \
                        "Loss_1:[{:.3f}]" "Loss_2:[{:.3f}]" "Loss_inf:[{:.3f}]"\
                        "Acc:[{:.3f}]".format(
                            epoch_i, i + 1, len(loader), loss_1.item(), loss_2.item(), loss_inf.item(), train_acc/train_n)

                log(p_str.format())
      
    return train_loss / train_n, train_acc / train_n, loss_1.item(), loss_2.item(), loss_inf.item()


def epoch_adversarial_union(args, loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
    opt=None, device = "cuda:0", stop = False, epsilon_l_1 = 12, epsilon_l_2 = 0.5, epsilon_l_inf = 0.03, stats = False, log = None, label_noises = 'False', **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss = 0
    train_acc1 = 0
    train_acc2 = 0
    train_acc3 = 0
    train_acc = 0
    train_n = 0
#     ipdb.set_trace()
    
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        ##Always calls the default version 0 for the individual attacks

        #L1
        delta = pgd_l1_topk(model, X, y, device = device, epsilon = epsilon_l_1,restarts = 10, num_iter = 100)
        output1 = model(X+delta)
        loss_1 = criterion(output1,y)
        train_loss1 += loss_1.item()*y.size(0)

        
        #L2
        delta = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2,restarts = 10, num_iter = 500)
        output2 = model(X+delta)
        loss_2 = nn.CrossEntropyLoss()(output2,y)
        train_loss2 += loss_2.item()*y.size(0)

        #Linf
        delta = pgd_linf(model, X, y, device = device, epsilon = epsilon_l_inf,restarts = 10, num_iter = 100)
        output3 = model(X+delta)
        loss_inf = nn.CrossEntropyLoss()(output2,y)
        train_loss3 += loss_inf.item()*y.size(0)

        train_acc1 +=  (output1.max(1)[1] == y).sum().item()
        train_acc2 +=  (output2.max(1)[1] == y).sum().item()         
        train_acc3 +=  (output3.max(1)[1] == y).sum().item() 

        train_acc +=  ((output1.max(1)[1] == y) & (output2.max(1)[1] == y) & (output3.max(1)[1] == y)).sum().item()
        train_n += y.size(0)
        
    return train_loss1 / train_n, train_loss2 / train_n, train_loss3 / train_n, train_acc1 / train_n, train_acc2 / train_n, train_acc3 / train_n, train_acc / train_n
