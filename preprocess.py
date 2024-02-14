import os 
import cv2 
import numpy as np 
from numba import njit, uint8
from skimage.measure import label, regionprops
import albumentations as A


def closing(image):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(image,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    return dilation


@njit(uint8[:, :](uint8[:, :]))
def recons_gris_bord_numba(Iin: uint8[:, :]) -> uint8[:, :]:
    Iout = np.full((Iin.shape[0], Iin.shape[1]), 255, np.uint8)
    Iout[1:-1, 1:-1] = 255
    Iout[0, :] = Iin[0, :]
    Iout[-1, :] = Iin[-1, :]
    Iout[:, 0] = Iin[:, 0]
    Iout[:, -1] = Iin[:, -1]

    # iteration aller-retour
    arret = True
    iter = 0
    while arret:
        arret = True
        iter += 1
        for i in range(1, Iin.shape[0]-1):
            for j in range(1, Iin.shape[1]-1):
                mn = min(Iout[i-1, j-1], Iout[i-1, j], Iout[i-1, j+1], Iout[i, j-1], Iout[i, j])
                if Iin[i, j] >= mn:
                    if Iout[i, j] != Iin[i, j]:
                        arret = False
                        Iout[i, j] = Iin[i, j]
                else:
                    if mn != Iout[i, j]:
                        arret = False
                        Iout[i, j] = mn
        for i in range(Iin.shape[0]-2, 0, -1):
            for j in range(Iin.shape[1]-2, 0, -1):
                mn = min(Iout[i+1, j+1], Iout[i+1, j], Iout[i+1, j-1], Iout[i, j+1], Iout[i, j])
                if Iin[i, j] >= mn:
                    if Iout[i, j] != Iin[i, j]:
                        arret = False
                        Iout[i, j] = Iin[i, j]
                else:
                    if mn != Iout[i, j]:
                        arret = False
                        Iout[i, j] = mn
                        
    return Iout
    
def get_larget_component(image):
    image_rec = recons_gris_bord_numba(image)
    image_rec = cv2.threshold(image_rec, 100, 255, cv2.THRESH_BINARY)[1]
    labeled_image = label(image_rec, connectivity=1, background=0)
    regions = regionprops(labeled_image)
    largest_area = 0
    largest_component = None

    for region in regions:
        if region.area > largest_area:
            largest_area = region.area
            largest_component = region

    bbox = largest_component.bbox

    return image[bbox[0]:bbox[2], bbox[1]:bbox[3]]


       
def get_cropped_bone(image): 
    image_th = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)[1]
    image_th = closing(image_th)
    # find index of the first and last non-zero values 4 coordinates of the bounding box
    (x,y) = np.where(image_th != 0)
    (top_x, top_y) = (np.min(x), np.min(y))
    (bottom_x, bottom_y) = (np.max(x), np.max(y))
    out = image[top_x:bottom_x+1, top_y:bottom_y+1]

    return out

def get_lung(image):    
    image = get_larget_component(image)
    image = get_cropped_bone(image)
    transform = A.Compose([
        A.Resize(256,256)
    ])
    aug = transform(image=image)
    image = aug['image']

    return image





        
