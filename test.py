import argparse, os
import torch
from model import UNet, ResUNet, UNet_LRes, ResUNet_LRes
from utils import DataLoaderVal
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description="PyTorch InfantSeg")

parser.add_argument("--gpuID", type=int, default=0, help="how to normalize the data")
parser.add_argument("--img_size", type=int, default=256, help="size of image")
parser.add_argument("--dataset", action="store_true", help="name of dataset", default='MRI-PET')
parser.add_argument("--numOfChannel_allSource", type=int, default=3, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--modelName", default="/home/niedong/Data4LowDosePET/pytorch_UNet/model/resunet2d_dp_pet_BatchAug_sNorm_lres_bn_lr5e3_lrdec_base1_lossL1_0p005_0628_200000.pt", type=str, help="modelname")
parser.add_argument("--inputKey", default="MRI", type=str, help="input modality")
parser.add_argument("--targetKey", default="PET", type=str, help="target modality")

global opt 
opt = parser.parse_args()

def main():
    
    os.makedirs('result', exist_ok=True)

    if opt.whichNet==1:
        netG = UNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==2:
        netG = ResUNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==3:
        netG = UNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==4:
        netG = ResUNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)

    #netG.apply(weights_init)
    netG.cuda()

    checkpoint = torch.load(os.path.join('checkpoints', opt.modelName))
    netG.load_state_dict(checkpoint['model'])

    test_dataset = DataLoaderVal(os.path.join('../dataset', opt.dataset, 'test'), opt.inputKey, opt.targetKey, {'w': opt.img_size, 'h': opt.img_size})
    testloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

    for i, data in enumerate(tqdm(testloader)):
        inputs = data[0]
        f_name = data[2][0]

        source = inputs
        
        #source = inputs
        mid_slice = opt.numOfChannel_singleSource//2
        residual_source = inputs[:, mid_slice, ...]

        source = source.cuda()
        residual_source = residual_source.cuda()
        labels = labels.cuda()

        if opt.whichNet == 3 or opt.whichNet == 4:
            outputG = netG(source, residual_source)  # 5x64x64->1*64x64
        else:
            outputG = netG(source)  # 5x64x64->1*64x64
        
        save_image(outputG, os.path.join('result', f_name))
        



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID) 
    main()
