# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.optim as optim
import torch
from utils import DataLoaderTrain, weights_init
from model import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from nnBuildUnits import RelativeThreshold_RegLoss, gdl_loss, adjust_learning_rate, calc_gradient_penalty
from torch.utils.data import DataLoader
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--gpuID", type=int, default=0, help="how to normalize the data")
parser.add_argument("--img_size", type=int, default=128, help="size of image")
parser.add_argument("--lambda_AD", default=0.05, type=float, help="weight for AD loss, Default: 0.05")
parser.add_argument("--lambda_D_WGAN_GP", default=10, type=float, help="weight for gradient penalty of WGAN-GP, Default: 10")
parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--gdlNorm", default=2, type=int, help="p-norm for the gdl loss, Default: 2")
parser.add_argument("--lambda_gdl", default=0.05, type=float, help="Weight for gdl loss, Default: 0.05")
parser.add_argument("--whichNet", type=int, default=4, help="which loss to use: 1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 3)")
parser.add_argument("--lossBase", type=int, default=1, help="The base to multiply the lossG_G, Default (1)")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--dataset", action="store_true", help="name of dataset", default='MRI-PET')
parser.add_argument("--numOfChannel_singleSource", type=int, default=3, help="# of channels for a 2D patch for the main modality (Default, 5)")
parser.add_argument("--numOfChannel_allSource", type=int, default=3, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--numofEpochs", type=int, default=100, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=10, help="number of iterations to save the model")
parser.add_argument("--lr", type=float, default=5e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_netD", type=float, default=5e-3, help="Learning Rate for discriminator. Default=5e-3")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="prob to drop neurons to zero: 0.2")
parser.add_argument("--decLREvery", type=int, default=10000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--lrDecRate", type=float, default=0.5, help="The weight for decreasing learning rate of netG Default=0.5")
parser.add_argument("--lrDecRate_netD", type=float, default=0.1, help="The weight for decreasing learning rate of netD. Default=0.1")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default=False, type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--RT_th", default=0.005, type=float, help="Relative thresholding: 0.005")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--inputKey", default="MRI", type=str, help="input modality")
parser.add_argument("--targetKey", default="PET", type=str, help="target modality")
parser.add_argument("--prefixModelName", default="temp", type=str, help="prefix of the to-be-saved model name")

global opt, model 
opt = parser.parse_args()

def main():    

    os.makedirs('checkpoints', exist_ok=True)

    netD = Discriminator()
    netD.apply(weights_init)
    netD.cuda()
    
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_netD)
    criterion_bce=nn.BCELoss()
    criterion_bce.cuda()
    
    #net=UNet()
    if opt.whichNet==1:
        net = UNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==2:
        net = ResUNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==3:
        net = UNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==4:
        net = ResUNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1, dp_prob = opt.dropout_rate)
    #net.apply(weights_init)
    net.cuda()
 
    
    optimizer = optim.Adam(net.parameters(),lr=opt.lr)
    criterion_L2 = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_RTL1 = RelativeThreshold_RegLoss(opt.RT_th)
    criterion_gdl = gdl_loss(opt.gdlNorm)
    
    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_RTL1 = criterion_RTL1.cuda()
    criterion_gdl = criterion_gdl.cuda()
    
    

    train_dataset = DataLoaderTrain(os.path.join('../dataset', opt.dataset, 'train'), opt.inputKey, opt.targetKey, {'w': opt.img_size, 'h': opt.img_size})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['model'])
            opt.start_epoch = 100000
            opt.start_epoch = checkpoint["epoch"] + 1
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


    for iter in range(opt.start_epoch, opt.numofEpochs+1):

        for i, data in enumerate(tqdm(trainloader)):
            inputs = data[0]
            labels = data[1]

            source = inputs
        
            #source = inputs
            mid_slice = opt.numOfChannel_singleSource//2
            residual_source = inputs[:, mid_slice, ...]

            source = source.cuda()
            residual_source = residual_source.cuda()
            labels = labels.cuda()
            
            #outputG = net(source,residual_source) #5x64x64->1*64x64
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
                
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
            
            outputD_real = netD(labels)
            outputD_real = torch.sigmoid(outputD_real)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)
            outputD_fake = torch.sigmoid(outputD_fake)
            netD.zero_grad()
            batch_size = inputs.size(0)
            real_label = torch.ones(batch_size,1)
            real_label = real_label.cuda()

            loss_real = criterion_bce(outputD_real,real_label)
            loss_real.backward()
            #train with fake data
            fake_label = torch.zeros(batch_size,1)
    #         fake_label = torch.FloatTensor(batch_size)
    #         fake_label.data.resize_(batch_size).fill_(0)
            fake_label = fake_label.cuda()
            loss_fake = criterion_bce(outputD_fake,fake_label)
            loss_fake.backward()

            optimizerD.step()
                
            one = torch.tensor(1)
            mone = one * -1
            one = one.cuda()
            mone = mone.cuda()
            
            netD.zero_grad()
            
            #outputG = net(source,residual_source) #5x64x64->1*64x64
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
                
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
                
            outputD_real = netD(labels)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)

            
            batch_size = inputs.size(0)
            
            D_real = outputD_real.mean()
            # print D_real
            D_real.backward(mone)
        
        
            D_fake = outputD_fake.mean()
            D_fake.backward(one)
        
            gradient_penalty = opt.lambda_D_WGAN_GP * calc_gradient_penalty(netD, labels.data, outputG.data)
            gradient_penalty.backward()
            
            optimizerD.step()
            
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source) 
            else:
                outputG = net(source) 

            net.zero_grad()
            if opt.whichLoss==1:
                lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))
            elif opt.whichLoss==2:
                lossG_G = criterion_RTL1(torch.squeeze(outputG), torch.squeeze(labels))
            else:
                lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(labels))
            lossG_G = opt.lossBase * lossG_G
            lossG_G.backward(retain_graph=True) #compute gradients

            lossG_gdl = opt.lambda_gdl * criterion_gdl(outputG,torch.unsqueeze(torch.squeeze(labels,1),1))
            lossG_gdl.backward() #compute gradients


            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  
            else:
                outputG = net(source) 
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD = netD(outputG)
            outputD = torch.sigmoid(outputD)
            lossG_D = opt.lambda_AD * criterion_bce(outputD,real_label) 
            lossG_D.backward()
                

            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source) 
            else:
                outputG = net(source)
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD_fake = netD(outputG)

            outputD_fake = outputD_fake.mean()
            
            lossG_D = opt.lambda_AD*outputD_fake.mean()
            lossG_D.backward(mone)
            
            
            optimizer.step() 

            if iter % opt.saveModelEvery==0:
                state = {
                    'epoch': iter+1,
                    'model': net.state_dict()
                }
                torch.save(state, os.path.join('checkpoints', opt.prefixModelName+'%d.pt'%iter))

                torch.save(netD.state_dict(), os.path.join('checkpoints', opt.prefixModelName+'_net_D%d.pt'%iter))

            if iter % opt.decLREvery==0:
                opt.lr = opt.lr*opt.lrDecRate
                adjust_learning_rate(optimizer, opt.lr)
                opt.lr_netD = opt.lr_netD*opt.lrDecRate_netD
                adjust_learning_rate(optimizerD, opt.lr_netD)
    
if __name__ == '__main__':   
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)  
    main()
    
