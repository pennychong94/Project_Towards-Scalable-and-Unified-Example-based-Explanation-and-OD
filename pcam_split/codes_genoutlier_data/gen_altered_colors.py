
import numpy as np
import os
import argparse
from PIL import Image
import random
import math
import colorsys
import h5py

random.seed(1)


def run(image_size, textfile_path, prefix_image, new_direc):
    # with open(textfile_path) as f:
    #   content = f.readlines()
    # content = [x.strip().split()[0] for x in content]

    fullh5name=os.path.join(prefix_image,textfile_path)
    data_x=h5py.File(fullh5name, "r")['x']
    num_images=data_x.shape[0]

    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    for c in range(num_images):
        im_np=data_x[c]
        print(im_np.shape) #(96, 96, 3)
        im=Image.fromarray(im_np)

        # img_name=os.path.join(prefix_image,c)
        # im = Image.open(img_name).convert('RGB')
        im = np.array(np.asarray(im).astype('float')) ## convert to numpy array
        #print(im.shape) #height x width x channel
        im = im/255.0
        r, g, b = np.rollaxis(im, axis=-1)
        h, s, v = rgb_to_hsv(r, g, b)  ## input must be in the range of 0-1, hsv output is also in the range of 0-1. for h, multiply with 360 degree to get real values

        s=np.maximum(s, 0.3*np.ones_like(s))
        v=np.maximum(v, 0.3*np.ones_like(v))

        z=np.random.randint(low=120,high=240,size=(image_size,image_size), dtype=np.uint8)
        h_new = np.int_(np.round(h*360))
        h_new = (h_new + z) % 360
        h = h_new/360.0
        r, g, b = hsv_to_rgb(h, s, v) ## input & output is all in the range of 0-1
        im_new = np.dstack((r, g, b))
        im_new = Image.fromarray(np.uint8(im_new*255))

        newImagename=os.path.join(new_direc,'testsubsetdataoutlier_'+str(c)+'.png')
        print(newImagename)
        dir_='/'.join(newImagename.split("/")[0:-1])
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        im_new.save(newImagename)





if __name__=='__main__':

    #python gen_altered_colors.py --dataset patchcam --rootPath /rootpath/to/new_allcls_patchcam_split/


    parser = argparse.ArgumentParser(description='generate synthetic altered color images')
    parser.add_argument('--dataset',type=str ,help='the class can be patchcam')
    parser.add_argument('--rootPath', type=str, help= 'the path to the file patch cam subset h5 /rootpath/to/new_allcls_patchcam_split/')


    args = parser.parse_args()


    assert args.dataset in ['patchcam']

    if args.dataset=='patchcam':
        textfile_ls=['new_allcls_patchcam_split/camelyonpatch_level_2_split_test_subset1500_x.h5'] ### dont need y labels for thus, use the 1500 per class one for outlier
        image_size=96 ## for patch cam

        prefix_image=args.rootPath
        new_direc="../../patchcam_synthetic/altered_colors"


    for txtfile_ in textfile_ls:
        textfile_path= txtfile_
        if not os.path.exists(new_direc):
            os.makedirs(new_direc)
        run(image_size, textfile_path, prefix_image, new_direc)
