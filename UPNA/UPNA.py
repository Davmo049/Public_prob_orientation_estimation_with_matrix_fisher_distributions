import os
import shutil

import scipy.io
import skimage
import skimage.transform
from PIL import Image
import imageio
import numpy as np
import torch

import matplotlib.pyplot as plt

PREPREPROCESSED_DIR = 'upna_preprocessed'
RAW_DIR = 'UPNA/Head_Pose_Database_UPNA'

# duplicated code Pascal3d
def get_back_proj_bbx(bbx, intrinsic):
    assert(len(bbx) == 4)
    minx = bbx[0]
    miny = bbx[1]
    maxx = bbx[2]
    maxy = bbx[3]
    points = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]])
    points = points.transpose()
    points_homo = np.ones((3,4))
    points_homo[:2, :] = points
    intrinsic_inv = np.linalg.inv(intrinsic)
    backproj = np.matmul(intrinsic_inv, points_homo)
    backproj /= np.linalg.norm(backproj, axis=0).reshape(1, -1)
    return backproj



# duplicated code from Pascal3D
def get_desired_camera(desired_imagesize, backproj, desired_up):
    z, radius_3d = get_minimum_covering_sphere(backproj.transpose())
    y = desired_up - np.dot(desired_up, z)*z
    y /= np.linalg.norm(y)
    x = np.cross(y,z)
    R = np.stack([x, y,z], axis=0)# AXIS? TODO
    bp_reproj = np.matmul(R, backproj)
    bp_reproj/=bp_reproj[2,:].reshape(1, -1)
    f = 1/np.max(np.abs(bp_reproj[:2]).reshape(-1))

    intrinsic = np.array([[desired_imagesize*f/2, 0, desired_imagesize/2],
                          [0, desired_imagesize*f/2, desired_imagesize/2],
                          [0, 0, 1]])
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = R
    return extrinsic, intrinsic

# duplicated code from Pascal3D
def get_minimum_covering_sphere(points):
    # points = nx3 array on unit sphere
    # returns point on unit sphere which minimizes the maximum distance to point in points
    # uses modified version of welzl
    points = np.copy(points)
    np.random.shuffle(points)
    def sphere_welzl(points, included_points, num_included_points):
        if len(points) == 0 or num_included_points == 3:
            return sphere_trivial(included_points[:num_included_points])
        else:
            p = points[0]
            rem = points[1:]
            cand_mid, cand_rad = sphere_welzl(rem, included_points, num_included_points)
            if np.linalg.norm(p-cand_mid) < cand_rad:
                return cand_mid, cand_rad
            included_points[num_included_points] = p
            return sphere_welzl(rem, included_points, num_included_points+1)
    buf = np.empty((3,3), dtype=np.float)
    return sphere_welzl(points, buf, 0)

# duplicated code
def sphere_trivial(points):
    if len(points) == 0:
        return np.array([1.0, 0,0]), 0
    elif len(points) == 1:
        return points[0], 0
    elif len(points) == 2:
        mid = (points[0] + points[1])/2
        diff = points-mid.reshape(1, -1)
        r = np.max(np.linalg.norm(diff, axis=1))
        return mid, r
    elif len(points) == 3:
        X = np.stack(points, axis=0)
        C = np.array([1,1,1])
        mid = np.linalg.solve(X, C)
        mid /= np.linalg.norm(mid)
        r = np.max(np.linalg.norm(points-mid.reshape(1, -1), axis=1))
        return mid, r
    raise Exception("2d welzl should not need 4 points")





def parse_csv(filename):
    with open(filename, 'r') as f:
        lines = []
        while True:
            l = f.readline()
            if len(l) == 0:
                break
            row_splits = l.split('\t')
            row_splits = row_splits[:-1] # remove last blank line
            row_splits_no_ws = []
            for entry in row_splits:
                while entry[0] == ' ':
                    entry = entry[1:]
                row_splits_no_ws.append(entry)

            stuff = list(map(float, row_splits_no_ws))
            lines.append(stuff)
        return lines

def create_preprocessed_dir(raw_dir, out_dir, visualize=True):
    os.makedirs(out_dir)
    subjects = []
    for folder in os.listdir(raw_dir):
        if len(folder) > 4:
            if folder[:5] == 'User_':
                subjects.append(folder)
    src_calib = os.path.join(raw_dir, 'Camera_parameters.mat')
    dst_calib = os.path.join(out_dir, 'Camera_parameters.mat')
    shutil.copyfile(src_calib, dst_calib)
    subjects = sorted(subjects)
    for subject in subjects:
        raw_dir_subject = os.path.join(raw_dir, subject)
        out_dir_subject = os.path.join(out_dir, subject)
        os.makedirs(out_dir_subject)
        files = []
        for filename in os.listdir(raw_dir_subject):
            if filename[-4:] == '.mp4':
                files.append(filename)
        for filename in files:
            base_name = filename[:-4]
            filename_keypoint = os.path.join(raw_dir_subject, base_name + '_groundtruth2D.txt')
            filename_3d = os.path.join(raw_dir_subject, base_name + '_groundtruth3D.txt')
            filename_mp4 = os.path.join(raw_dir_subject, base_name + '.mp4')
            vid = imageio.get_reader(filename_mp4,  'ffmpeg')
            keypoints_data = parse_csv(filename_keypoint)

            def split_xy(coords):
                keypoints = []
                cur_point = None
                point_idx = 0
                for coord in coords:
                    if cur_point is None:
                        cur_point = np.empty(2)
                    cur_point[point_idx] = coord
                    point_idx += 1
                    if point_idx == 2:
                        keypoints.append(cur_point)
                        point_idx = 0
                        cur_point = None
                return keypoints
            keypoints = list(map(split_xy, keypoints_data))


            info_3d = parse_csv(filename_3d)
            for frame_idx, (image, kp_frame, frame_3d) in enumerate(zip(vid, keypoints, info_3d)):
                if frame_idx % 20 != 0:
                    continue
                kp_array = np.array(kp_frame)
                bbx_min_x = np.min(kp_array[:, 0])
                bbx_max_x = np.max(kp_array[:, 0])
                bbx_min_y = np.min(kp_array[:, 1])
                bbx_max_y = np.max(kp_array[:, 1])
                center_x = (bbx_min_x+bbx_max_x)/2
                center_y = (bbx_min_y+bbx_max_y)/2
                width = (bbx_max_x - bbx_min_x + 1)*1.6 - 1
                height = (bbx_max_y - bbx_min_y + 1)*1.6 - 1

                # permute center and width and height slightly to remove detailed keypoint info from bbx
                center_x += np.random.uniform(-0.1, 0.1)*width
                center_y += np.random.uniform(-0.1, 0.1)*height

                width *= np.random.uniform(0.9, 1.1)
                height *= np.random.uniform(0.9, 1.1)

                bbx_min_x = int(round(center_x - width/2))
                bbx_max_x = int(round(center_x + width/2))
                bbx_min_y = int(round(center_y - height/2))
                bbx_max_y = int(round(center_y + height/2))
                bbx = [bbx_min_x, bbx_min_y, bbx_max_x, bbx_max_y]
                rotation_angles = np.array(frame_3d[3:]) # roll yaw pitch
                def get_rotation_from_angles(angles):
                    # input is roll yaw pitch in degrees
                    angles *= np.pi/180
                    cx = np.cos(angles[0])
                    sx = np.sin(angles[0])

                    cy = np.cos(-angles[1])
                    sy = np.sin(-angles[1])

                    cz = np.cos(angles[2])
                    sz = np.sin(angles[2])

                    Rx = np.array([[cx, sx, 0], [-sx, cx, 0], [0,0,1]])
                    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy,0,cy]])
                    Rz = np.array([[1, 0, 0], [0, cz, sz], [0,-sz, cz]])
                    R = np.matmul(Rx, np.matmul(Ry, Rz))
                    return R
                rotation = get_rotation_from_angles(rotation_angles)

                # save frame
                rotation_file = os.path.join(out_dir_subject, 'rotation_{}_{}.txt'.format(base_name, frame_idx))
                bbx_file = os.path.join(out_dir_subject, 'bbx_{}_{}.txt'.format(base_name, frame_idx))
                image_file = os.path.join(out_dir_subject, 'image_{}_{}.png'.format(base_name, frame_idx))

                PIL_im = Image.fromarray(image)
                PIL_im.save(image_file)
                with open(rotation_file, 'w') as f:
                    for row_idx in range(3):
                        row = rotation[row_idx]
                        f.write('{} {} {}\n'.format(row[0], row[1], row[2]))

                with open(bbx_file, 'w') as f:
                    f.write('{} {} {} {}'.format(bbx[0], bbx[1], bbx[2], bbx[3]))

                # visualize
                if visualize:
                    # show image
                    plt.imshow(image)
                    # show key points
                    plt.imshow(image)
                    for j in range(len(kp_frame)):
                        k = kp_frame[j]
                        plt.plot(k[0], k[1], 'rx')
                    # show bbx
                    x_plot_bbx = [bbx[0], bbx[2], bbx[2], bbx[0], bbx[0]]
                    y_plot_bbx = [bbx[1], bbx[1], bbx[3], bbx[3], bbx[1]]
                    plt.plot(x_plot_bbx, y_plot_bbx, 'b')
                    # show axis
                    f = 1400/1.5
                    ppx = np.array([990, 570])/1.5
                    distance_approx = 4
                    center_3d = np.ones(3)
                    center_3d[:2] = (kp_frame[31] - ppx)/f
                    center_3d *= distance_approx
                    points = np.zeros((4,3))
                    points[1:] = np.matmul(np.array([[1.0, 0, 0], [0, -1, 0], [0,0,-1]]), rotation)
                    points += center_3d.reshape(1, 3)
                    points_2d = points[:, :2] / points[:, 2].reshape(4,1)
                    points_2d = points_2d*f+ppx.reshape(1,2)
                    for axis_idx in range(3):
                        color = ['r', 'g', 'b'][axis_idx]
                        point_start = points_2d[0]
                        point_end = points_2d[axis_idx+1]

                        plt.plot([point_start[0], point_end[0]], [point_start[1], point_end[1]], color)
                    plt.show()
            assert(len(keypoints))
            assert(len(info_3d))
            if (frame_idx != len(keypoints)-1):
                print(filename_mp4 + ' Does not have expected number of frames')
                exit(0)



class UPNA():
    def __init__(self, dataset_path, desired_image_size=224):
        self.dataset_preprocessed = os.path.join(dataset_path, PREPREPROCESSED_DIR)
        remove_existing_preprocessed = False
        if remove_existing_preprocessed and os.path.exists(self.dataset_preprocessed):
            shutil.rmtree(self.dataset_preprocessed)
        if not os.path.exists(self.dataset_preprocessed):
            dataset_raw = os.path.join(dataset_path, RAW_DIR)
            create_preprocessed_dir(dataset_raw, self.dataset_preprocessed)
        train_users = ['User_01', 'User_02', 'User_03', 'User_04', 'User_05', 'User_06']
        val_users = ['User_07', 'User_08', 'User_09', 'User_10']
        def get_samples_for_users(base_folder, users):
            ret = []
            for user in users:
                user_folder = os.path.join(base_folder, user)
                for filename in os.listdir(user_folder):
                    if filename[-4:] == '.png':
                        ret.append((user, filename[6:-4]))
            return ret
        self.train_samples = sorted(get_samples_for_users(self.dataset_preprocessed, train_users))
        self.val_samples = sorted(get_samples_for_users(self.dataset_preprocessed, val_users))
        def get_calibration(base_folder):
            mat_data = scipy.io.loadmat(os.path.join(self.dataset_preprocessed, 'Camera_parameters.mat'))
            ret = {} # maybe not the prettiest
            ret['focal_length'] = mat_data['focal_length'].reshape(2) / 1.5 # suspect video images are scaled version of calibration images
            ret['principal_point'] = mat_data['principal_point'].reshape(2) / 1.5
            return ret

        calibration = get_calibration(self.dataset_preprocessed)
        self.focal_length = calibration['focal_length']
        self.principal_point = calibration['principal_point']
        self.desired_image_size = desired_image_size

    def get_train(self):
        return self.get_upna_subset(self.train_samples, True)

    def get_eval(self):
        return self.get_upna_subset(self.val_samples, False)

    def get_upna_subset(self, samples, aug):
        return UpnaSubset(self, samples, aug)

class UpnaSubset(torch.utils.data.Dataset):
    def __init__(self, full_set, samples, aug):
        self.full_set = full_set
        self.samples = samples
        self.aug = aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user, filename = sample
        subject_dir = os.path.join(self.full_set.dataset_preprocessed, user)
        image_file = os.path.join(subject_dir, 'image_' + filename + '.png')
        rotation_file = os.path.join(subject_dir, 'rotation_' + filename + '.txt')
        bbx_file = os.path.join(subject_dir, 'bbx_' + filename + '.txt')
        img_PIL = Image.open(image_file)
        img_full = np.array(img_PIL.getdata()).reshape(img_PIL.size[1], img_PIL.size[0], 3)
        bbox = np.empty(4)
        with open(bbx_file, 'r') as f:
            line = f.readline()
            vals = list(map(float, line.split(' ')))
            for i in range(4):
                bbox[i] = vals[i]
        extrinsic = np.eye(4)
        with open(rotation_file, 'r') as f:
            for r in range(3):
                line = f.readline()
                line = line[:-1]
                vals = list(map(float, line.split(' ')))
                for c in range(3):
                    extrinsic[r,c] = vals[c]
        extrinsic = extrinsic.transpose()
        principal_point = np.copy(self.full_set.principal_point)
        if self.aug:
            flip = np.random.randint(2)
            if flip==1:
                # flip x then flip x axis
                extrinsic[:3, :3] = np.array([-1, 1,1]).reshape(1, 3)*extrinsic[:3,:3]*np.array([-1, 1,1]).reshape(3, 1)
                img_full = img_full[:, ::-1, :]
                bbox[0] = img_full.shape[1] - bbox[0]
                bbox[2] = img_full.shape[1] - bbox[2]
                principal_point[0] = img_full.shape[1] - principal_point[0]
            # # change up direction of warp
            desired_up = np.random.normal(0,0.7,size=(3))+np.array([0.0, 3.0, 0.0])
            desired_up[2] = 0
            desired_up /= np.linalg.norm(desired_up)
            # # jitter bounding box
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox[0::2] += np.random.uniform(-bbox_w*0.1, bbox_w*0.1, size=(2))
            bbox[1::2] += np.random.uniform(-bbox_h*0.1, bbox_h*0.1, size=(2))
        else:
            desired_up = np.array([0.0, 1.0, 0.0])

        intrinsic = np.array([
            [self.full_set.focal_length[0], 0, self.full_set.principal_point[0]],
            [0, self.full_set.focal_length[1], self.full_set.principal_point[1]],
            [0, 0, 1]])

        back_proj_bbx = get_back_proj_bbx(bbox, intrinsic)

        desired_imagesize = self.full_set.desired_image_size
        extrinsic_desired_change, intrinsic_new = get_desired_camera(desired_imagesize, back_proj_bbx, desired_up)
        extrinsic_after = np.matmul(extrinsic_desired_change, np.array([1,-1,-1,1]).reshape(1,4) * extrinsic)
        P = np.matmul(np.matmul(intrinsic_new, extrinsic_desired_change[:3, :3]), np.linalg.inv(intrinsic))
        P /= P[2,2]
        Pinv = np.linalg.inv(P)
        transform = skimage.transform.ProjectiveTransform(Pinv)
        im = img_full.astype(np.float)/255
        warped_image = skimage.transform.warp(im, transform, output_shape=(desired_imagesize, desired_imagesize), mode='constant', cval=0.0)
        extrinsic_after[2,3] = 5

        return warped_image.transpose(2,0,1).astype(np.float32), extrinsic_after.astype(np.float32), 0, False, intrinsic_new.astype(np.float32), 0



def main():
    dataset_dir = 'datasets' # TODO change to where datasets are stored
    x = UPNA(dataset_dir)
    ds = x.get_train()
    for idx_i in range(28, len(ds)):
        idx = idx_i*50
        sample = ds[idx]
        image, extrinsic, class_idx, hard, intrinsic, cad_idx = sample
        print(intrinsic)
        print(extrinsic[:3,:3])
        print(extrinsic)
        image = image.transpose(1,2,0)
        points = np.array([[0.0,0,0,1],
                           [1,0,0,1],
                           [0,1,0,1],
                           [0,0,1,1]]).transpose()
        points = np.matmul(extrinsic, points)
        points = points/points[2]
        points = points[:3, :]
        points = np.matmul(intrinsic, points)
        plt.imshow(image)
        print(points)
        for p, c in zip(points[:,1:].transpose(), ['r','g','b']):
            x = [points[0,0], p[0]]
            y = [points[1,0], p[1]]
            plt.plot(x,y,c,linewidth=5)
        plt.savefig('images/UPNA_example.pdf')
        exit(0)



if __name__ == '__main__':
    main()
