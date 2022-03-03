import numpy as np
import SimpleITK as sitk

import os
import pdb 

def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage

def ResampleFullImageToRef(imImage, imRef, interp=sitk.sitkNearestNeighbor):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    imRefImage = sitk.Image(imRef.GetSize(), imImage.GetPixelIDValue())
    imRefImage.SetSpacing(imRef.GetSpacing())
    imRefImage.SetOrigin(imRef.GetOrigin())
    imRefImage.SetDirection(imRef.GetDirection())


    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage


def ResampleCMRImage(imImage, imLabel, save_path, patient_name, name, target_space=(1., 1.)):
    
    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()


    npimg = sitk.GetArrayFromImage(imImage)
    nplab = sitk.GetArrayFromImage(imLabel)
    t, z, y, x = npimg.shape
    
    if not os.path.exists('%s/%s'%(save_path, patient_name)):
        os.mkdir('%s/%s'%(save_path, patient_name))
    flag = 0
    for i in range(t):
        tmp_img = npimg[i]
        tmp_lab = nplab[i]
        
        if tmp_lab.max() == 0:
            continue

        
        print(i)
        flag += 1
        tmp_itkimg = sitk.GetImageFromArray(tmp_img)
        tmp_itkimg.SetSpacing(spacing[0:3])
        tmp_itkimg.SetOrigin(origin[0:3])
            
        tmp_itklab = sitk.GetImageFromArray(tmp_lab)
        tmp_itklab.SetSpacing(spacing[0:3])
        tmp_itklab.SetOrigin(origin[0:3])

        
        re_img = ResampleXYZAxis(tmp_itkimg, space=(target_space[0], target_space[1], spacing[2]))
        re_lab = ResampleFullImageToRef(tmp_itklab, re_img)

        
        sitk.WriteImage(re_img, '%s/%s/%s_%d.nii.gz'%(save_path, patient_name, name, flag))
        sitk.WriteImage(re_lab, '%s/%s/%s_gt_%d.nii.gz'%(save_path, patient_name, name, flag))
        
    return flag


if __name__ == '__main__':
    

    src_path = 'OpenDataset/Training/Labeled'
    tgt_path = 'dataset/Training'

    os.chdir(src_path)
    for name in os.listdir('.'):
        os.chdir(name)

        for i in os.listdir('.'):
            if 'gt' in i:
                tmp = i.split('_')
                img_name = tmp[0] + '_' + tmp[1]
                patient_name = tmp[0]
                
                img = sitk.ReadImage('%s.nii.gz'%img_name)
                lab = sitk.ReadImage('%s_gt.nii.gz'%img_name)
                
                flag = ResampleCMRImage(img, lab, tgt_path, patient_name, img_name, (1.2, 1.2))
                
                print(name, 'done', flag)

        os.chdir('..')




