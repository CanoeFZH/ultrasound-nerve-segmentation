import pandas as pd
import numpy as np
import glob

def RLencToPixels(runs):
    p1 = [] # Run-start pixel locations
    p2 = [] # Run-lengths
    
    # Separate run-lengths and pixel locations into seperate lists
    x = str(runs).split(' ')
    i = 0
    for m in x:
        if i % 2 == 0:
            p1.append(m)
        else:
            p2.append(m)
        i += 1
        
    # Get all absolute pixel values
    pixels = []
    for start, length in zip(p1, p2):
        i = 0
        length = int(length)
        pix = int(start)
        while i < length:
            pixels.append(pix)
            pix += 1
            i += 1
            
    return pixels

import pylab as plt
import RLE


if __name__ == '__main__':



    subs = [img for img in glob.glob("ensemble/*.csv")]
    new_sub = np.zeros((5508,420,580)).astype(np.int8)
    total = 0
    print np.float32(total)
    for sub in subs:
        data = pd.read_csv(sub)['pixels'].fillna('').values
        count = 0
        for img in data:

            # print type(img)
            # print img

            origin_mask = RLencToPixels(img)
            tmp = np.zeros(420*580).astype(np.int8)
            # print origin_mask
            origin_mask = [i-1 for i in origin_mask]
            tmp[origin_mask]=1
            tmp = tmp.reshape(580,420).T
            # plt.imshow(tmp)
            # plt.show()
            new_sub[count,:,:]+=tmp.astype(np.int8)
            # print new_sub.shape
            # print new_sub[count,:,:]
            count+=1
        total+=1
    
    print('Averaging')
    new_sub = new_sub.astype(np.float32)
    new_sub/=np.float32(total)
    new_sub[new_sub>0.0]=1
    new_sub[new_sub<=0.0]=0
    np.save("imgs_mask_test.npy",np.expand_dims(new_sub,1))

    new_sub = np.load("imgs_mask_test.npy")
    print new_sub.shape
    for img in new_sub:
        plt.imshow(img[0])
        plt.show()
