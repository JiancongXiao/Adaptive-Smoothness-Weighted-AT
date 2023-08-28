import sys
sys.path.append('./models/')
import torch
import time
from models import PreActResNet18
from models import PreActResNet18_silu
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import ipdb
import sys 
import argparse
sys.path.append('./utils/')
from core import *
from torch_backend import *
from cifar_funcs import *
from swa_utils import *
from datetime import datetime
from tensorboardX import SummaryWriter


# python3 train.py -gpu_id 0 -model 3 -batch_size 128 -lr_schedule 1
parser = argparse.ArgumentParser(description='Adversarial Training for CIFAR10', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("--model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla \n\t 7: mix \n\t 8: sat \n\t 9: adt", type=int, default = 9)
parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 128)", type = int, default = 128)
parser.add_argument("--swa", type = bool, default = True, help = 'using stochastic weight averaging')
parser.add_argument("--swat", type = bool, default = False, help = 'using stochastic weight averaging training')
parser.add_argument('--swa_start', type=float, default=60, metavar='N', help='SWA start epoch number (default: 20)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N', help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='Training epochs (default: 50)')
parser.add_argument('--exp_id', type=str, default='t', metavar='N', help='The name of Experiments')
parser.add_argument('--test', type=bool, default=True, help='train or test')
parser.add_argument('--print_freq', type=int, default=5, metavar='N', help='print frequency')
parser.add_argument('--silu', type=bool, default=False, help='relu or silu')
parser.add_argument('--label_smoothing', type=str, default='False', help='smoothing')
parser.add_argument('--label_noise', type=str, default='False', help='noise')
parser.add_argument('--seed', type=int, default=1, help='noise')
args = parser.parse_args()
device_id = args.gpu_id
ln = args.label_noise
timestamp = str(datetime.now())[:-7]


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

torch.cuda.device_count() 
batch_size = args.batch_size
choice = args.model

epochs = args.epochs
DATA_DIR = '../data'
dataset = cifar10(DATA_DIR)

train_set = list(zip(transpose(normalise2(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])
train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2, gpu_id = torch.cuda.current_device())
test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())

# import pdb; pdb.set_trace()

if args.silu:
    model = PreActResNet18_silu().cuda()
else:
    model = PreActResNet18().cuda()

for m in model.children(): 
    if not isinstance(m, nn.BatchNorm2d):
        m.half()  

if args.swa:
    if args.silu:
        swa_model = PreActResNet18_silu().cuda()
    else:
        swa_model = PreActResNet18().cuda()

    swa_n = 0 
    for m in swa_model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.half() 


opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

iif args.label_smoothing == 'True':
    criterion = LabelSmoothing(smoothing = 0.1)
else:
    criterion = nn.CrossEntropyLoss()

import time

lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.005, 0])[0]
# lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.1, 0.1])[0]
args.swa_start = epochs*3//5
#For clearing pytorch cuda inconsistency
try:
    train_loss, train_acc = epoch(test_batches, lr_schedule, model, 0, criterion, opt = None, device = device, stop = True)
except:
    a =1

attack_list = [ pgd_linf,  pgd_l1_topk,   pgd_l2 ,  msd_v0 ,  triple_adv ,  pgd_worst_dir, triple_adv, mix, sat, adt] #TRIPLE, VANILLA DON'T HAVE A ATTACK NAME ANYTHING WORKS
attack_name = ["pgd_linf", "pgd_l1_topk", "pgd_l2", "msd_v0", "triple_adv", "pgd_worst_dir", "vanilla", "mix", 'sat', 'adt']
folder_name = ["LINF", "L1", "L2", "MSD_V0", "TRIPLE", "WORST", "VANILLA", "MIX", 'SAT', 'ADT']

if args.silu == True:
    exp_name = folder_name[choice]+'_'+ args.exp_id+'_'+'silu'
else:
    exp_name = folder_name[choice]+'_'+ args.exp_id

model_dir = "Final100/{0}".format(exp_name)
import os
if(not os.path.exists(model_dir)):
    os.makedirs(model_dir)

file = open("{0}/logs.txt".format(model_dir), "w")


def myprint(a):
    print(a)
    file.write(a)
    file.write("\n")

attack = attack_list[choice]

hps_str = '{} model={} swa={} swa_start={} swa_c_epochs={} epochs={} test={} '.format(
    timestamp, args.model, args.swa, args.swa_start, args.swa_c_epochs, args.epochs, args.test)

myprint('All hps: {}'.format(hps_str))
myprint((attack_name[choice]+'_'+ args.exp_id).format())

train_loss_list = torch.tensor([])
swa_train_loss_list = torch.tensor([])
l1_list = torch.tensor([])
swa_l1_list = torch.tensor([])
l2_list = torch.tensor([])
swa_l2_list = torch.tensor([])
linf_list = torch.tensor([])
swa_linf_list = torch.tensor([])
clean_list = torch.tensor([])
swa_clean_list = torch.tensor([])
train_acc_list = torch.tensor([])
swa_train_acc_list = torch.tensor([])
loss_l1_list = torch.tensor([])
loss_l2_list = torch.tensor([])
loss_linf_list = torch.tensor([])

for epoch_i in range(1,epochs+1):  

    if epoch_i == 1:
        model_eval = model
    start_time = time.time()
    lr = lr_schedule(epoch_i + (epoch_i+1)/len(train_batches))
    if choice == 6:
        train_epoch = epoch
        train_loss, train_acc = epoch(args, train_batches, lr_schedule, model, epoch_i, criterion, opt = opt, device = device, log = myprint, )
    elif choice == 4:
        train_epoch = triple_adv
        train_loss, train_acc,loss_l1, loss_l2, loss_linf = triple_adv(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.5, log = myprint, label_noises = ln)
    elif choice == 3:
        train_epoch = epoch_adversarial
        train_loss, train_acc = epoch_adversarial(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.5, log = myprint, label_noises = ln)
    elif choice == 7:
        train_epoch = epoch_adversarial_mix
        train_loss, train_acc = epoch_adversarial_mix(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.5, log = myprint)
    elif choice == 8:
        train_epoch = epoch_adversarial_sat
        train_loss, train_acc = epoch_adversarial_sat(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.5, log = myprint, label_noises = ln)
    elif choice == 5:
        train_epoch = epoch_adversarial
        train_loss, train_acc = epoch_adversarial(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.5, alpha_l_inf = 0.005, log = myprint, label_noises = ln)
    elif choice == 9:
        train_epoch = epoch_adversarial_adt
        train_loss, train_acc, loss_l1, loss_l2, loss_linf = epoch_adversarial_adt(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_2 = 0.5, log = myprint)
    else:
        train_epoch = epoch_adversarial
        train_loss, train_acc = epoch_adversarial(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device, log = myprint)


    if args.swa and epoch_i >= args.swa_start and (epoch_i - args.swa_start) % args.swa_c_epochs == 0:
        myprint('using SWA'.format())
        moving_average(swa_model, model, alpha=1.0 / (swa_n + 1))
        swa_n += 1
        bn_update(train_batches, swa_model)
        # torch.save(swa_model.state_dict(),"{0}/swa_iter_{1}.pth".format(model_dir, str(epoch_i)))
        model_eval = swa_model
        myprint('Finishing SWA'.format())
        if choice == 4 or choice == 9: 
            train_loss_all, train_acc_all,_ ,_ ,_ = train_epoch(args, train_batches, lr_schedule, model_eval, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
            train_loss, train_acc,_ ,_ ,_ = train_epoch(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
        else:  
            train_loss_all, train_acc_all = train_epoch(args, train_batches, lr_schedule, model_eval, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
            train_loss, train_acc = train_epoch(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
    
    else:
        model_eval = model
        if choice == 4 or choice == 9: 
            train_loss_all, train_acc_all,_ ,_ ,_ = train_epoch(args, train_batches, lr_schedule, model_eval, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
            train_loss, train_acc,_ ,_ ,_ = train_epoch(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
        else:  
            train_loss_all, train_acc_all = train_epoch(args, train_batches, lr_schedule, model_eval, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
            train_loss, train_acc = train_epoch(args, train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = None, device = device, stop = True, log = myprint)
    res = 10
    
    if args.test: 
        total_loss, total_acc   = epoch(args, test_batches, lr_schedule, model_eval, epoch_i, criterion, opt = None, device = device, log = myprint)
        total_loss, total_acc_1 = epoch_adversarial(args, test_batches, None,  model_eval, epoch_i, pgd_l1_topk,criterion,device = device, stop = True, restarts = res, num_iter = 40, log = myprint)
        total_loss, total_acc_2 = epoch_adversarial(args, test_batches, None, model_eval, epoch_i, pgd_l2, criterion,device = device, stop = True, restarts = res, epsilon = 0.5, num_iter = 40, alpha = 0.01, log = myprint)   
        total_loss, total_acc_3 = epoch_adversarial(args, test_batches, None, model_eval, epoch_i, pgd_linf, criterion,device = device, stop = True, restarts = res, num_iter = 40, log = myprint)

        total_loss_cur, total_acc_cur   = epoch(args, test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = device, log = myprint)
        total_loss, total_acc_1_cur = epoch_adversarial(args, test_batches, None, model, epoch_i, pgd_l1_topk,criterion,device = device, stop = True, restarts = res, num_iter = 40, log = myprint)
        total_loss, total_acc_2_cur = epoch_adversarial(args, test_batches, None, model, epoch_i, pgd_l2, criterion,device = device, stop = True, restarts = res, epsilon = 0.5, num_iter = 40, alpha = 0.01, log = myprint)
        total_loss, total_acc_3_cur = epoch_adversarial(args, test_batches, None, model, epoch_i, pgd_linf, criterion,device = device, stop = True, restarts = res, num_iter = 40, log = myprint)

    else:
        total_loss, total_acc   = epoch(args, test_batches, lr_schedule, model_eval, epoch_i, criterion, opt = None, device = device, log = myprint)
        total_loss, total_acc_1 = epoch_adversarial(args, test_batches, lr_schedule, model_eval, epoch_i,  pgd_l1_topk, criterion, opt = None, device = device, stop = True, log = myprint)
        total_loss, total_acc_2 = epoch_adversarial(args, test_batches, lr_schedule, model_eval, epoch_i,  pgd_l2, criterion, opt = None, device = device, stop = True, log = myprint)
        total_loss, total_acc_3 = epoch_adversarial(args, test_batches, lr_schedule, model_eval, epoch_i,  pgd_linf, criterion, opt = None, device = device, stop = True, log = myprint)

        total_loss_cur, total_acc_cur   = epoch(args, test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = device, log = myprint)
        total_loss_cur, total_acc_1_cur = epoch_adversarial(args, test_batches, lr_schedule, model, epoch_i,  pgd_l1_topk, criterion, opt = None, device = device, stop = True, log = myprint)
        total_loss_cur, total_acc_2_cur = epoch_adversarial(args, test_batches, lr_schedule, model, epoch_i,  pgd_l2, criterion, opt = None, device = device, stop = True, log = myprint)
        total_loss_cur, total_acc_3_cur = epoch_adversarial(args, test_batches, lr_schedule, model, epoch_i,  pgd_linf, criterion, opt = None, device = device, stop = True, log = myprint)

    myprint('Epoch: {7}, Clean Acc: {6:.4f} Train Acc: {5:.4f}, Test Acc 1: {4:.4f}, Test Acc 2: {3:.4f}, Test Acc inf: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_3, total_acc_2,total_acc_1,train_acc, total_acc, epoch_i))
    myprint('Epoch: {7}, Clean Acc: {6:.4f} Train Acc: {5:.4f}, Test Acc 1_cur: {4:.4f}, Test Acc 2_cur: {3:.4f}, Test Acc inf_cur: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_3_cur, total_acc_2_cur,total_acc_1_cur,train_acc, total_acc_cur, epoch_i))    
    myprint('Epoch: {6}, train loss: {5:.4f} train acc: {4:.4f}, train loss all: {3:.4f}, train acc all: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, train_acc_all,train_loss_all,train_acc, train_loss, epoch_i))

    train_loss_list = torch.cat((train_loss_list,torch.tensor([train_loss])), dim = 0)
    swa_train_loss_list = torch.cat((swa_train_loss_list,torch.tensor([train_loss_all])), dim = 0)
    l1_list = torch.cat((l1_list,torch.tensor([total_acc_1_cur])), dim = 0)
    swa_l1_list = torch.cat((swa_l1_list,torch.tensor([total_acc_1])), dim = 0)
    l2_list = torch.cat((l2_list,torch.tensor([total_acc_2_cur])), dim = 0)
    swa_l2_list = torch.cat((swa_l2_list,torch.tensor([total_acc_2])), dim = 0)
    linf_list = torch.cat((linf_list,torch.tensor([total_acc_3_cur])), dim = 0)
    swa_linf_list = torch.cat((swa_linf_list,torch.tensor([total_acc_3])), dim = 0)
    clean_list = torch.cat((clean_list,torch.tensor([total_acc_cur])), dim = 0)
    swa_clean_list = torch.cat((swa_clean_list,torch.tensor([total_acc])), dim = 0)
    train_acc_list = torch.cat((train_acc_list,torch.tensor([train_acc])), dim = 0)
    swa_train_acc_list = torch.cat((swa_train_acc_list,torch.tensor([train_acc_all])), dim = 0)
    if choice == 4 or choice == 9:    
        loss_l1_list = torch.cat((loss_l1_list,torch.tensor([loss_l1])), dim = 0)
        loss_l2_list = torch.cat((loss_l2_list,torch.tensor([loss_l2])), dim = 0)
        loss_linf_list = torch.cat((loss_linf_list,torch.tensor([loss_linf])), dim = 0)
        np.save('{}/loss_l1_list.npy'.format(model_dir),loss_l1_list.numpy())
        np.save('{}/loss_l2_list.npy'.format(model_dir),loss_l2_list.numpy())
        np.save('{}/loss_linf_list.npy'.format(model_dir),loss_linf_list.numpy())

    np.save('{}/train_loss_list.npy'.format(model_dir),train_loss_list.numpy())
    np.save('{}/swa_train_loss_list.npy'.format(model_dir),swa_train_loss_list.numpy())
    np.save('{}/l1_list.npy'.format(model_dir),l1_list.numpy())
    np.save('{}/swa_l1_list.npy'.format(model_dir),swa_l1_list.numpy())
    np.save('{}/l2_list.npy'.format(model_dir),l2_list.numpy())
    np.save('{}/swa_l2_list.npy'.format(model_dir),swa_l2_list.numpy())
    np.save('{}/linf_list.npy'.format(model_dir),linf_list.numpy())
    np.save('{}/swa_linf_list.npy'.format(model_dir),swa_linf_list.numpy())
    np.save('{}/clean_list.npy'.format(model_dir),clean_list.numpy())
    np.save('{}/swa_clean_list.npy'.format(model_dir),swa_clean_list.numpy())
    np.save('{}/train_acc_list.npy'.format(model_dir),train_acc_list.numpy())
    np.save('{}/swa_train_acc_list.npy'.format(model_dir),swa_train_acc_list.numpy())
    
 #   if train_loss <=0.7:
  #      break    
Loss_1, Loss_2, Loss_3, acc_1, acc_2, acc_3, acc_u = epoch_adversarial_union(args, test_batches, None, model, pgd_linf, criterion,device = device, stop = True, restarts = res, num_iter = 100, log = myprint)
myprint('orignal model,l1 acc: {0:4f},l2 acc: {1:4f},linf acc: {2:4f}, average acc: {3:4f}, union acc: {4:.4f}'.format(acc_1, acc_2, acc_3,(acc_1+acc_2+acc_3)/3, acc_u))
Loss_1, Loss_2, Loss_3, acc_1, acc_2, acc_3, acc_u = epoch_adversarial_union(args, test_batches, None, model_eval, pgd_linf, criterion,device = device, stop = True, restarts = res, num_iter = 100, log = myprint)
myprint('swa model, l1 acc: {0:4f},l2 acc: {1:4f},linf acc: {2:4f}, average acc: {3:4f}, union acc: {4:.4f}'.format(acc_1, acc_2, acc_3,(acc_1+acc_2+acc_3)/3, acc_u))

torch.save(model.state_dict(), "{0}/{1}.pth".format(model_dir,args.exp_id))
torch.save(swa_model.state_dict(),"{0}/swa_{1}.pth".format(model_dir,args.exp_id))




