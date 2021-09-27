import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import pdb 


def log_evaluation_result(writer, dice_list, ASD_list, HD_list, name, epoch):
    
    writer.add_scalar('Test_Dice/%s_AVG'%name, dice_list.mean(), epoch+1)
    for idx in range(3):
        writer.add_scalar('Test_Dice/%s_Dice%d'%(name, idx+1), dice_list[idx], epoch+1)
    writer.add_scalar('Test_ASD/%s_AVG'%name, ASD_list.mean(), epoch+1)
    for idx in range(3):
        writer.add_scalar('Test_ASD/%s_ASD%d'%(name, idx+1), ASD_list[idx], epoch+1)
    writer.add_scalar('Test_HD/%s_AVG'%name, HD_list.mean(), epoch+1)
    for idx in range(3):
        writer.add_scalar('Test_HD/%s_HD%d'%(name, idx+1), HD_list[idx], epoch+1)


def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr

def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr



def cal_dice(pred, target, C): 
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.cuda()
    dice = 2 * intersection / summ

    return dice, intersection, summ

def cal_asd(itkPred, itkGT):
    
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(itkGT, squaredDistance=False))
    reference_surface = sitk.LabelContour(itkGT)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(itkPred, squaredDistance=False))
    segmented_surface = sitk.LabelContour(itkPred)

    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0])
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0])
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    
    all_surface_distances = seg2ref_distances + ref2seg_distances

    ASD = np.mean(all_surface_distances)

    return ASD

