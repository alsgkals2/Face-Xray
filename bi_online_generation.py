import dlib
from skimage import io
from skimage import transform as sktransform
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import random
from PIL import Image
from imgaug import augmenters as iaa
from DeepFakeMask import dfl_full,facehull,components,extended
import cv2
import tqdm

def name_resolve(path):
    name_folder = path.split('/')[-2]
    name = path.split('/')[-1]
    name= name_folder+'_'+name
    vid_id = name.split('.jpg')[0]
    return vid_id, None

def total_euclidean_distance(a,b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a-b,axis=1))

def random_get_hull(landmark,img1):
    hull_type = random.choice([0,1,2,3])
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255

def random_erode_dilate(mask, ksize=None):
    if random.random()>0.5:
        if ksize is  None:
            ksize = random.randint(1,21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.erode(mask,kernel,1)/255
    else:
        if ksize is  None:
            ksize = random.randint(1,5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.dilate(mask,kernel,1)/255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):
    maskIndices = np.where(mask != 0)
    
    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    
    maskIndices = np.where(mask != 0)
    

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst

class BIOnlineGeneration():
    def __init__(self,path_json, save_path):
        self.save_path= save_path
        with open(path_json, 'r') as f:
            self.landmarks_record =  json.load(f)
            for i,(k,v) in enumerate(self.landmarks_record.items()):
                self.landmarks_record[k] = np.array(v)

        self.list_path=[]
        for a,b,c in os.walk(self.save_path):
            for _c in c:
                if '.jpg' in _c:
                    self.list_path.append(os.path.join(a,_c))
        print(self.list_path)
        self.data_list = list(self.landmarks_record.keys())
        
        # predefine mask distortion
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        
    def gen_one_datapoint(self,idx=-1):
        title = None
        if idx==-1:
            background_face_path = random.choice(self.data_list)
        else:
            background_face_path = self.data_list[idx]
            title = self.data_list[idx].split('/')[-2]
            title += '_' + self.data_list[idx].split('/')[-1]
        if os.path.join(self.save_path,title) in self.list_path:
            print("already exist")
            return None,None,None,None

        data_type = 'fake'
        # data_type = 'real' if random.randint(0,1) else 'fake'
        if data_type == 'fake' :
            face_img,mask =  self.get_blended_face(background_face_path)
            if mask is None:
                return None,None,None,None
            mask = ( 1 - mask ) * mask * 4

        return face_img,mask,data_type, title
        
    def get_blended_face(self,background_face_path):
        background_face = io.imread(background_face_path)
        background_landmark = self.landmarks_record[background_face_path]
        
        foreground_face_path = self.search_similar_face(background_landmark,background_face_path)
        foreground_face = io.imread(foreground_face_path)
        
        # down sample before blending
        aug_size = random.randint(128,317)
        background_landmark = background_landmark * (aug_size/317)
        foreground_face = sktransform.resize(foreground_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        background_face = sktransform.resize(background_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        
        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)
       
        #  random deform mask
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)
        
        # filte empty mask after deformation
        if np.sum(mask) == 0 :
            print('mask is empty!!!!!!!!!')
            print(f'background_face_path : {background_face_path}')
            return None,None
            # raise NotImplementedError

        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask*255)
        
        # blend two face
        blended_face, mask = blendImages(foreground_face, background_face, mask*255)
        blended_face = blended_face.astype(np.uint8)
       
        # resize back to default resolution
        blended_face = sktransform.resize(blended_face,(317,317),preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask,(317,317),preserve_range=True)
        mask = mask[:,:,0:1]
        return blended_face,mask
    
    def search_similar_face(self,this_landmark,background_face_path):
        min_dist = 99999999
        num_sample = 1000
        # random sample frame from all frams:
        all_candidate_path = random.sample( self.data_list, k=num_sample)
        
        # filter all frame that comes from the same video as background face
        all_candidate_path = list(all_candidate_path)
        
        # loop throungh all candidates frame to get best match
        for candidate_path in all_candidate_path:
            candidate_landmark = self.landmarks_record[candidate_path].astype(np.float32)
            candidate_distance = total_euclidean_distance(candidate_landmark, this_landmark)
            if candidate_distance < min_dist:
                min_dist = candidate_distance
                min_path = candidate_path

        return min_path

    def check_aleady_exist(self, path):
        if path in self.list_path:
            print("already exist")
            return True
        else:
            return False

if __name__ == '__main__':
    save_path = 'FRAME 이미지 들어있는 경로'
    ds = BIOnlineGeneration('랜드마크 정보있는 json 파일', save_path)
    from tqdm import tqdm
    all_imgs = []
    cnt =0
    idx_half = len(ds.data_list)//2
    print('before' , len(ds.data_list))
    ds.data_list = ds.data_list[:idx_half]
    print('after' , len(ds.data_list))
    num_iter = len(ds.data_list)
    for i,_ in tqdm(enumerate(range(num_iter))):
        img,mask,label,title = ds.gen_one_datapoint(i)
        if mask is None:
            print('CAN NOT WORK BECAUSE OF EMPTY MASK ')
            continue
        mask = np.repeat(mask,3,2)
        mask = (mask*255).astype(np.uint8)
        img = Image.fromarray(img).resize((224,224))
        img.save(f'{save_path}/{title}')