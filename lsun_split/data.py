'''
This code is adapted from Fisher Yu to resize image before saving it in PNG format.
The  targetsize variable is the size of the image after resizing. It is set to 128.
'''

from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy
import os
from os.path import exists, join

import PIL.Image

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


import codecs

def slashescape(err):
    """ codecs error handler. err is UnicodeDecode instance. return
    a tuple with a replacement for the unencodable part of the input
    and a position where encoding should continue"""
    #print err, dir(err), err.start, err.end, err.object[:err.start]
    thebyte = err.object[err.start:err.end]
    repl = u'\\x'+hex(ord(thebyte))[2:]
    return (repl, err.end)

codecs.register_error('slashescape', slashescape)

def view(db_path):
    print('Viewing', db_path)
    print('Press ESC to exist or SPACE to advance.')
    window_name = 'LSUN'
    cv2.namedWindow(window_name)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            print('Current key:', key)
            img = cv2.imdecode(
                numpy.fromstring(val, dtype=numpy.uint8), 1)
            cv2.imshow(window_name, img)
            c = cv2.waitKey()
            if c == 27:
                break


def export_images(db_path, out_dir, flat=False, limit=-1):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            #key=key.decode("utf-8")
            if not flat:
                print(out_dir,key[:6])
                image_out_dir = join(out_dir, '/'.join(key[:6]))
            else:
                image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key + '.webp')
            #print(val)
            with open(image_out_path, 'w') as fp:
                fp.write(val) #.decode('utf-8', 'backslashreplace'))

            #resize
            targetsize=128



            oriimg = cv2.imread(image_out_path)
            height, width, depth = oriimg.shape
            imgScale = targetsize/float(min(height, width))
            newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
            if newX < newY:
              newX=targetsize
            else:
              newY=targetsize

            newX=int(newX)
            newY=int(newY)

            print( width, height,depth,int(newX),int(newY))

            newimg = cv2.resize(oriimg,(int(newX),int(newY)))
            print(oriimg.shape, 'newimg.shape', newimg.shape)
            newimg=newimg[int((newY-targetsize)/2):int((newY-targetsize)/2)+targetsize,int((newX-targetsize)/2):int((newX-targetsize)/2)+targetsize,:]
            print(oriimg.shape, 'newimg2.shape', newimg.shape)
            #cv2.imshow("Show by CV2",newimg)
            #cv2.waitKey(0)
            os.remove(image_out_path)
            cv2.imwrite(image_out_path[:-5]+'.png',newimg)
            #cv2.imwrite(image_out_path[:-5]+'.webp',newimg)


            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', type=str,
                        choices=['view', 'export'],
                        help='view: view the images in the lmdb database '
                             'interactively.\n'
                             'export: Export the images in the lmdb databases '
                             'to a folder. The images are grouped in subfolders'
                             ' determinted by the prefiex of image key.')
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--flat', action='store_true',
                        help='If enabled, the images are imported into output '
                             'directory directly instead of hierarchical '
                             'directories.')
    args = parser.parse_args()

    command = args.command
    lmdb_paths = args.lmdb_path

    for lmdb_path in lmdb_paths:
        if command == 'view':
            view(lmdb_path)
        elif command == 'export':
            export_images(lmdb_path, args.out_dir+'/'+lmdb_path.split('/')[-1], args.flat, limit=10000)


if __name__ == '__main__':
    main()
