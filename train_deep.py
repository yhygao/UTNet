import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from model.utnet import UTNet, UTNet_Encoderonly

from dataset_domain import CMRDataset

from torch.utils import data
from losses import DiceLoss
from utils.utils import *
from utils import metrics
from optparse import OptionParser
import SimpleITK as sitk

from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False

def train_net(net, options):
    
    data_path = options.data_path

    trainset = CMRDataset(data_path, mode='train', domain=options.domain, debug=DEBUG, scale=options.scale, rotate=options.rotate, crop_size=options.crop_size)
    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=16)

    testset_A = CMRDataset(data_path, mode='test', domain='A', debug=DEBUG, crop_size=options.crop_size)
    testLoader_A = data.DataLoader(testset_A, batch_size=1, shuffle=False, num_workers=2)
    testset_B = CMRDataset(data_path, mode='test', domain='B', debug=DEBUG, crop_size=options.crop_size)
    testLoader_B = data.DataLoader(testset_B, batch_size=1, shuffle=False, num_workers=2)
    testset_C = CMRDataset(data_path, mode='test', domain='C', debug=DEBUG, crop_size=options.crop_size)
    testLoader_C = data.DataLoader(testset_C, batch_size=1, shuffle=False, num_workers=2)
    testset_D = CMRDataset(data_path, mode='test', domain='D', debug=DEBUG, crop_size=options.crop_size)
    testLoader_D = data.DataLoader(testset_D, batch_size=1, shuffle=False, num_workers=2)





    writer = SummaryWriter(options.log_path + options.unique_name)

    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(options.weight).cuda())
    criterion_dl = DiceLoss()


    best_dice = 0
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0

        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, max_epoch=options.epochs)

        print('current lr:', exp_scheduler)

        for i, (img, label) in enumerate(trainLoader, 0):

            img = img.cuda()
            label = label.cuda()

            end = time.time()
            net.train()

            optimizer.zero_grad()
            
            result = net(img)
            
            loss = 0
            
            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    loss += options.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)


            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()
            batch_time = time.time() - end
            print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', exp_scheduler, epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        if epoch % 20 == 0 or epoch > options.epochs-10:
            torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch))
        
        if (epoch+1) >90 or (epoch+1) % 10 == 0:
            dice_list_A, ASD_list_A, HD_list_A = validation(net, testLoader_A, options)
            log_evaluation_result(writer, dice_list_A, ASD_list_A, HD_list_A, 'A', epoch)
            
            dice_list_B, ASD_list_B, HD_list_B = validation(net, testLoader_B, options)
            log_evaluation_result(writer, dice_list_B, ASD_list_B, HD_list_B, 'B', epoch)

            dice_list_C, ASD_list_C, HD_list_C = validation(net, testLoader_C, options)
            log_evaluation_result(writer, dice_list_C, ASD_list_C, HD_list_C, 'C', epoch)

            dice_list_D, ASD_list_D, HD_list_D = validation(net, testLoader_D, options)
            log_evaluation_result(writer, dice_list_D, ASD_list_D, HD_list_D, 'D', epoch)


            AVG_dice_list = 20 * dice_list_A + 50 * dice_list_B + 50 * dice_list_C + 50 * dice_list_D
            AVG_dice_list /= 170

            AVG_ASD_list = 20 * ASD_list_A + 50 * ASD_list_B + 50 * ASD_list_C + 50 * ASD_list_D
            AVG_ASD_list /= 170

            AVG_HD_list = 20 * HD_list_A + 50 * HD_list_B + 50 * HD_list_C + 50 * HD_list_D
            AVG_HD_list /= 170

            log_evaluation_result(writer, AVG_dice_list, AVG_ASD_list, AVG_HD_list, 'mean', epoch)



            if dice_list_A.mean() >= best_dice:
                best_dice = dice_list_A.mean()
                torch.save(net.state_dict(), '%s%s/best.pth'%(options.cp_path, options.unique_name))

            print('save done')
            print('dice: %.5f/best dice: %.5f'%(dice_list_A.mean(), best_dice))


def validation(net, test_loader, options):

    net.eval()
    
    dice_list = np.zeros(3)
    ASD_list = np.zeros(3)
    HD_list = np.zeros(3)

    counter = 0
    with torch.no_grad():
        for i, (data, label, spacing) in enumerate(test_loader):
            
            inputs, labels = data.float().cuda(), label.long().cuda()
            inputs = inputs.permute(1, 0, 2, 3)
            labels = labels.permute(1, 0, 2, 3)
    
            pred = net(inputs)
            if options.model == 'FCN_Res50' or options.model == 'FCN_Res101':
                pred = pred['out']
            elif isinstance(pred, tuple):
                pred = pred[0]
            pred = F.softmax(pred, dim=1)
    
            _, label_pred = torch.max(pred, dim=1)
            
            tmp_ASD_list, tmp_HD_list = cal_distance(label_pred, labels, spacing)
            ASD_list += np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            HD_list += np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)
            
            label_pred = label_pred.view(-1, 1)
            label_true = labels.view(-1, 1)

            dice, _, _ = cal_dice(label_pred, label_true, 4)
        
            dice_list += dice.cpu().numpy()[1:]
            counter += 1
    dice_list /= counter
    avg_dice = dice_list.mean()
    ASD_list /= counter
    HD_list /= counter

    return dice_list , ASD_list, HD_list




def cal_distance(label_pred, label_true, spacing):
    label_pred = label_pred.squeeze(1).cpu().numpy()
    label_true = label_true.squeeze(1).cpu().numpy()
    spacing = spacing.numpy()[0]

    ASD_list = np.zeros(3)
    HD_list = np.zeros(3)

    for i in range(3):
        tmp_surface = metrics.compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
        dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2

        HD = metrics.compute_robust_hausdorff(tmp_surface, 100)
        HD_list[i] = HD

    return ASD_list, HD_list




if __name__ == '__main__':
    parser = OptionParser()
    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)

    parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=32, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.05, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/', help='checkpoint path')
    parser.add_option('--data_path', type='str', dest='data_path', default='/research/cbim/vast/yg397/vision_transformer/dataset/resampled_dataset/', help='dataset path')

    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='UTNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=4, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='test', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5,1,1,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=256)
    parser.add_option('--domain', type='str', dest='domain', default='A')
    parser.add_option('--aux_weight', type='float', dest='aux_weight', default=[1, 0.4, 0.2, 0.1])
    parser.add_option('--reduce_size', dest='reduce_size', default=8, type='int')
    parser.add_option('--block_list', dest='block_list', default='1234', type='str')
    parser.add_option('--num_blocks', dest='num_blocks', default=[1,1,1,1], type='string', action='callback', callback=get_comma_separated_int_args)
    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')

    
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    print('Using model:', options.model)

    if options.model == 'UTNet':
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
    elif options.model == 'UTNet_encoder':
        # Apply transformer blocks only in the encoder
        net = UTNet_Encoderonly(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
    elif options.model =='TransUNet':
        from model.transunet import VisionTransformer as ViT_seg
        from model.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 4 
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(256/16), int(256/16))
        net = ViT_seg(config_vit, img_size=256, num_classes=4)
        #net.load_from(weights=np.load('./initmodel/R50+ViT-B_16.npz')) # uncomment this to use pretrain model download from TransUnet git repo

    elif options.model == 'ResNet_UTNet':
        from model.resnet_utnet import ResNet_UTNet
        net = ResNet_UTNet(1, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True)
    
    elif options.model == 'SwinUNet':
        from model.swin_unet import SwinUnet, SwinUnet_config
        config = SwinUnet_config()
        net = SwinUnet(config, img_size=224, num_classes=options.num_class)
        net.load_from('./initmodel/swin_tiny_patch4_window7_224.pth')


    else:
        raise NotImplementedError(options.model + " has not been implemented")
    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(net)
    print(param_num)
    
    net.cuda()
    
    train_net(net, options)

    print('done')

    sys.exit(0)
