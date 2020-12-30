from numba import jit
import numpy as np
import cv2
from scipy import ndimage as ndi
import sys
import os


use_forward = True

def add_seam(img, seam_index):
    h, w = img.shape[:2]
    out = np.zeros((h,w+1,3))
    for i in range(h):
        j = int(seam_index[i])
        for t in range(3):
            if(j == 0):
                p = np.average(img[i,j:j+2,t])
                out[i,j,t] = img[i,j,t]
                out[i,j+1,t] = p
                out[i,j+1:,t] = img[i,j:,t]
            else:
                p = np.average(img[i,j-1:j+1,t])
                out[i,:j,t] = img[i,:j,t]
                out[i,j,t] = p
                out[i,j+1:,t] = img[i,j:,t]
    return out

def delete_seam(img,deleted_mask):
    h, w = img.shape[:2]

    deleted_mask = np.stack([deleted_mask] * 3, axis = 2)
    out = img[deleted_mask].reshape((h,w-1,3))
    return out

def delete_seam_mask(img,deleted_mask):
    h, w = img.shape[:2]

    out = img[deleted_mask].reshape((h,w-1))
    return out

def forward_energy(img):
    h, w = img.shape[:2]
    
    up = np.roll(img, 1, axis=0)
    left = np.roll(img, 1, axis=1)
    right = np.roll(img, -1, axis = 1)

    cu = np.sum(np.abs(right - left),axis=2)
    cl = np.sum(np.abs(up-left),axis=2)+cu
    cr = np.sum(np.abs(up-right),axis=2)+cu
    energy = np.zeros((h, w))
    p = np.zeros((h, w))
    for i in range(1,h):
        p_up = p[i-1]
        p_lft = np.roll(p_up,1)
        p_rht = np.roll(p_up,-1)
        
        p_stack = np.stack((p_up,p_lft,p_rht),axis=0)
        c_stack = np.stack((cu[i],cl[i],cr[i]),axis=0)
        p_stack += c_stack

        min_map = np.argmin(p_stack,axis=0)
        p[i] = np.choose(min_map,p_stack)
        energy[i] = np.choose(min_map,c_stack)

    return energy

def backward_energy(im): #gradient
    xgrad = np.abs(ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap'))
    ygrad = np.abs(ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap'))
    
    energy_map = np.sum(xgrad, axis=2) + np.sum(ygrad, axis=2)

    return energy_map


@jit
def get_min_cut(img, mask = None):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    
    global use_forward
    h, w = img.shape[:2]

    calc_energy = forward_energy if use_forward else backward_energy

    energy_map = calc_energy(img)

    if( mask is not None ):
        energy_map[np.where(mask >= 10)] = -100000000000.0

    former = np.zeros_like(energy_map,dtype=np.int)
    for i in range(1,h):
        for j in range(0,w):
            place = np.argmin(energy_map[i-1, max(0,j-1):j+2])
            former[i,j] = max(0,j-1)+place
            energy_map[i,j] += energy_map[i-1,max(0,j-1)+place]
    index = np.zeros(h,dtype = np.int)
    last = np.argmin(energy_map[-1])
    fast = np.ones((h,w),dtype= np.bool)
    for i in range(h-1,-1,-1):
        index[i] = last
        fast[i,last] = False
        last = former[i,last]

    return index, fast



def seams_removal(img,num_pixel):
    for _ in range(num_pixel):
        remove_index, deleted = get_min_cut(img)
        img = delete_seam(img,deleted)
    return img


def seams_insertion(img,num_pixel):
    seams = []
    temp_img = img.copy()
    
    for _ in range(num_pixel):
        remove_index , deleted = get_min_cut(temp_img)
        seams.append(remove_index)
        temp_img = delete_seam(temp_img,deleted)

    for _ in range(num_pixel):
        seam = seams.pop(0)
        img = add_seam(img,seam)
        for t in seams:
            t[np.where(t >= seam)] += 2
    return img    



def seam_carve(img,dy,dx):
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    out = img
    
    if(dx < 0):
        out = seams_removal(out,-dx)
    elif(dx > 0):
        out = seams_insertion(out,dx)

    if(dy != 0):
        out = np.rot90(out,1)

        if(dy > 0):
            out = seams_insertion(out,dy)
        else:
            out = seams_removal(out, -dy)
        out = np.rot90(out,3)
    return out

def object_removal(img, mask):
    img = img.astype(np.float64)
    out = img.copy()
    mask = mask.astype(np.float64)[:,:,0]
    h, w = img.shape[:2]
    l = len(np.where(mask > 10)[0])
    while(len(np.where(mask > 10)[0]) > 0):
        _, deleted = get_min_cut(out, mask)
        out = delete_seam(out,deleted)
        mask = delete_seam_mask(mask,deleted)
    num_pixel = w - out.shape[1]
    out = seams_insertion(out ,num_pixel)
    return out
        

if __name__ == '__main__':
    folder = 'data'

    filename_in = sys.argv[1]
    filename_out = sys.argv[2]

    img = cv2.imread(os.path.join(folder,'in',filename_in))
    l = len(sys.argv)
    if(l == 5):
        h,w = img.shape[:2]
        h1 = int(sys.argv[3])
        w1 = int(sys.argv[4])
        out = seam_carve(img,h1-h,w1-w)
    else:
        msk = cv2.imread(os.path.join(folder,'in',sys.argv[3]))
        out = object_removal(img,msk)
    cv2.imwrite(os.path.join(folder,'out',filename_out),out)