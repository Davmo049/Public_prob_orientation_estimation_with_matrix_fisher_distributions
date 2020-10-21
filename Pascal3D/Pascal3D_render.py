import torch
import numpy
import glob
from Pascal3D import Pascal3D
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt

id_to_pascal_class = {
    '02691156': Pascal3D.PascalClasses.AEROPLANE,
    '02834778': Pascal3D.PascalClasses.BICYCLE,
    '02858304': Pascal3D.PascalClasses.BOAT,
    '02876657': Pascal3D.PascalClasses.BOTTLE,
    '02924116': Pascal3D.PascalClasses.BUS,
    '02958343': Pascal3D.PascalClasses.CAR,
    '03001627': Pascal3D.PascalClasses.CHAIR,
    '03211117': Pascal3D.PascalClasses.TVMONITOR,
    '03790512': Pascal3D.PascalClasses.MOTORBIKE,
    '04256520': Pascal3D.PascalClasses.SOFA,
    '04379243': Pascal3D.PascalClasses.DININGTABLE,
    '04468005': Pascal3D.PascalClasses.TRAIN
}

DATASET_FOLDER = 'syn_images_cropped_bkg_overlaid'
class Pascal3DRendered(Dataset):
    def __init__(self, path, image_size=224):
        self.path = os.path.join(path, DATASET_FOLDER)
        all_files = glob.glob(os.path.join(self.path, '*/*/*.jpeg'))
        relpaths = list(map(lambda x: x[len(self.path)+1:], all_files))
        self.relpaths = relpaths
        self.image_out_size = image_size

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, idx):
        relpath = self.relpaths[idx]
        str_components = relpath.split('/')
        file_str = str_components[-1]
        synsetID, shapeID, a,e,t,d = file_str.split('_')
        assert(a[0]=='a' and e[0]=='e' and t[0]=='t')
        a,e,t = map(lambda x: float(x[1:]), [a,e,t])
        # aug start
        img_full = Image.open(os.path.join(self.path, relpath))
        img_full = np.array(img_full.getdata()).reshape(img_full.size[1], img_full.size[0],3).astype(np.float) / 255
        current_size = max(img_full.shape[1], img_full.shape[0])
        bbox = [0, 0, img_full.shape[1], img_full.shape[0]]
        distance = 4.0
        a = a
        e = e
        t = t
        cam = 3000
        principal_point = np.array([img_full.shape[1]/2, img_full.shape[0]/2], dtype=np.float)

        flip = np.random.randint(2)
        if flip:
            a = -a
            t = -t
            img_full = img_full[:, ::-1, :]
            bbox[0] = img_full.shape[1] - bbox[0]
            bbox[2] = img_full.shape[1] - bbox[2]
            principal_point[0] = img_full.shape[1] - principal_point[0]

        # # change up direction of warp
        desired_up = np.array([3.0, 0.0, 0.0]) + np.random.normal(0,0.4,size=(3))
        desired_up[2] = 0
        desired_up /= np.linalg.norm(desired_up)
        # # jitter bounding box
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox[0::2] += np.random.uniform(-bbox_w*0.1, bbox_w*0.1, size=(2))
        bbox[1::2] += np.random.uniform(-bbox_h*0.1, bbox_h*0.1, size=(2))

        angle = np.array([a,e,-t])
        intrinsic, extrinsic = Pascal3D.get_camera_parameters(cam, principal_point, angle, img_full.shape, distance)
        back_proj_bbx = Pascal3D.get_back_proj_bbx(bbox, intrinsic)
        desired_imagesize=self.image_out_size
        extrinsic_desired_change, intrinsic_new = Pascal3D.get_desired_camera(desired_imagesize, back_proj_bbx, desired_up)
        extrinsic_after = np.matmul(extrinsic_desired_change, extrinsic)

        P = np.matmul(np.matmul(intrinsic_new, extrinsic_desired_change[:3, :3]), np.linalg.inv(intrinsic))
        P /= P[2,2]
        Pinv = np.linalg.inv(P)
        transform = skimage.transform.ProjectiveTransform(Pinv)

        warped_image = skimage.transform.warp(img_full, transform, output_shape=(desired_imagesize, desired_imagesize), mode='constant', cval=0.0)

        class_enum = id_to_pascal_class[synsetID]

        return warped_image.transpose(2,0,1).astype(np.float32), extrinsic_after.astype(np.float32), int(class_enum), False, intrinsic_new.astype(np.float32), 0

def aet_to_R(a,e,t):
    a = a*np.pi/180
    e = e*np.pi/180
    t = t*np.pi/180
    t = -t

    a = -a
    e = e-np.pi/2

    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0,np.sin(e), np.cos(e)]])
    R2flip1 = np.array([[1.0, 0, 0], [0, 1, 0], [0,0,-1]])
    R2th = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0,0,1]])
    R2flip2 = np.array([[1.0, 0, 0], [0, -1, 0], [0,0,1]])
    R2 = np.matmul(R2flip2, np.matmul(R2th, R2flip1))
    R = np.matmul(R2, np.matmul(Rx, Rz))
    return R

def main():
    ds_path = 'datasets' # change to where datasets are stored
    ds = Pascal3DRendered(ds_path, 224)
    for i in range(100):
        im, ext, cls, hard, intr, cad = ds[i]
        im = im.transpose(1,2,0)
        points = np.array([[0.0,0,0,1],
                           [1,0,0,1],
                           [0,1,0,1],
                           [0,0,1,1]]).transpose()
        points = np.matmul(ext, points)
        points = points/points[2]
        points = points[:3, :]
        points = np.matmul(intr, points)
        plt.imshow(im)
        for p, c in zip(points[:,1:].transpose(), ['r','g','b']):
            x = [points[0,0], p[0]]
            y = [points[1,0], p[1]]
            plt.plot(x,y,c)
        plt.show()


if __name__ == '__main__':
    main()
