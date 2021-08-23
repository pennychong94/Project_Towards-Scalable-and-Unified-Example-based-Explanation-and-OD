## gen image with synthetic strokes
##must run with python3 (bcuz of np.block)

import numpy as np
import os
import argparse
from PIL import Image, ImageDraw
import random
import glob
import math
import json


random.seed(1)

def run(thickness, image_size, textfile_path, prefix_image, new_direc, seq_list):


    with open(textfile_path) as f:
      content = f.readlines()
    content = [x.strip().split()[0] for x in content]

    for c in content:
        rand_int=random.randint(5,10) # random int N such that 5 <= N <= 10.
        sub_seq_list=random.sample(seq_list, rand_int)
        gen_strokes(image_size, thickness, c, prefix_image, new_direc, sub_seq_list)


def gen_strokes(image_size, thickness, c, prefix_image, new_direc, sub_seq_list):
    img_name=os.path.join(prefix_image,c)
    im = Image.open(img_name).convert('RGB')
    im = im.resize((image_size,image_size), Image.ANTIALIAS)
    #im = np.asarray(im) ## convert to numpy array
    #print(im.shape) #height x width x channel
    print(img_name)
    dw = ImageDraw.Draw(im)

    make_color = lambda : (random.randint(0, 255), random.randint(0, 255), random.randint(0,255))
    selected_c=make_color() ## all the strokes in an image has the same color, but different images have diff stroke colors

    for seq in sub_seq_list:
        for i in range(1, len(seq)):
            last_point=seq[i-1]
            current_point=seq[i]
            dw.line([last_point[0], last_point[1], current_point[0], current_point[1]], fill=selected_c, width=thickness)

    #final_image = Image.fromarray(final_image)
    newImagename=os.path.join(new_direc,c)
    dir_='/'.join(newImagename.split("/")[0:-1])
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    im.save(newImagename)

def extract_mousemovement(mousePath, image_size, maxLength=30):

    json_data= open(mousePath, 'r')
    data = json.load(json_data)
    json_data.close()
    print("loaded train json data")

    seq_list=[]
    for item in data:
        width=item['width']
        height=item['height']
        seq=item['seq']
        newseq=transform_seq(width, height, seq, image_size)
        if len(newseq)>= maxLength:
            st=random.randint(0,len(newseq)-maxLength) ##both ends are inclusive
            seq_list.append(newseq[st:st+maxLength])
    return seq_list

def transform_seq(width, height, seq, image_size):

    newseq=[]
    for s in seq:
        if (0<= s[0]<= width) and (0<= s[1]<= height):
            newx=int((s[0]/float(width))*image_size)
            newy=int((s[1]/float(height))*image_size)
            newseq.append([newx,newy])
    return newseq



if __name__=='__main__':

    ##example command line
    ##python gen_synthetic_strokes.py --dataset lsun --thickness 5 --rootPath /some/path/to/LSUN_images_PL/ --mousePath standardized_balabit_twos/balabit/fused_train_images_1s_1.0_with_time_v2.json


    parser = argparse.ArgumentParser(description='generate synthetic strokes data')
    parser.add_argument('--dataset',type=str ,help='the class can be lsun')
    parser.add_argument('--thickness', type=int, help='thickness of the stroke')
    parser.add_argument('--rootPath', type=str, help= 'the root path to the file LSUN_images_PL')
    parser.add_argument('--mousePath', type=str, default='standardized_balabit_twos/balabit/fused_train_images_1s_1.0_with_time_v2.json' ,help= 'the path to the mouse movement json file')

    args = parser.parse_args()


    assert args.dataset in ['lsun']


    if args.dataset=='lsun':
        textfile_ls=['../new_allcls_lsun_split/new_test.txt']
        # textfile_ls=['../new_allcls_lsun_split/new10k_train.txt']
        image_size=128 ## for lsun

        prefix_image=os.path.join(args.rootPath,"LSUN_images_PL")
        new_direc="../../lsun_synthetic/strokes_"+str(args.thickness)



    seq_list=extract_mousemovement(args.mousePath, image_size)
    print("Len of filtered seq list :{}".format(len(seq_list)))

    for txtfile_ in textfile_ls:
        textfile_path= txtfile_
        if not os.path.exists(new_direc):
            os.makedirs(new_direc)
        run(args.thickness, image_size, textfile_path, prefix_image, new_direc, seq_list)
