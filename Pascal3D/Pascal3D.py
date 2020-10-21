import scipy.io
import copy
import wget
import os
import zipfile
from enum import IntEnum
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import skimage
import skimage.transform
import matplotlib.pyplot as plt
from .parse_off import parse_off_file

url = 'ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip'
DATASET_FOLDER_NAME = 'Pascal3d'
JSON_ANNOTATION_FOLDER_NAME = 'json_annotation'
TMP_ZIP_NAME = 'tmp.zip'



#format
#data['record']: {
#   'filename': string, filename
#   'folder': string
#   'source': struct for database etc
#   'imgname': string, path
#   'size': dimensions
#       'height'
#       'width'
#       'depth'
#   'segmented': 0/1 ?
#   'imgsize': dimensions [h,w,c]
#   'database': source database
#   'objects': list of objects
#       'class': string, class
#       'view': string, Frontal/Rear/Left/Right
#       'bbox': bounding box
#       'bndbox': bounding box as map (might only exist sometimes)
#       'orglabel': string includes a bunch of stuff (might only exist sometimes)
#       'truncated': 0/1
#       'occluded': 0/1
#       'difficult': 0/1
#       'anchors': list of anchors and their coordinates in image, contents depend on class, map between strings and anchors
#           'location': [], or [x,y] position in image
#           'status': something
#       'viewpoint': essentially angle
#           'azimuth_coarse':
#           'azimuth': 
#           'elevation_coarse':
#           'elevation':
#           'distance':
#           'px': center
#           'py': center
#           'theta':
#           'error':
#           'interval_azimuth':
#           'interval_elevation':
#           'num_anchor':
#           'viewport':
#       'cad_index': related to which cad was used
#       'polygon': empty list
#       'point': empty list
#       'part': empty list
#       'hasparts': 0/1
#       'actions': list
#       'hasactions': 0/1
#       'mask': 0/1
#  sometimes additional dimensions of size 1 is inserted probably due to matlab


def download_pascal3d(location):
    os.makedirs(location)
    tmp_zip_file = os.path.join(location, TMP_ZIP_NAME)
    wget.download(url, tmp_zip_file)
    with zipfile.ZipFile(tmp_zip_file, 'r') as zip_ref:
        zip_ref.extractall(location)
    # os.remove(tmp_zip_file)
    #TODO I have already downloaded dataset. make sure it works


def get_mat_element(data):
    while isinstance(data, np.ndarray):
        if len(data) == 0:
            raise PascalParseError("Encountered Empty List")
        if len(data) > 1:
            x = data[0]
            for y in data:
                if y != x:
                    print(data[0])
                    print(data[1])
                    raise(Exception("blah" + str(data)))
        data = data[0]
    return data


def get_mat_list(data):
    data_old = data
    while len(data) == 1:
        data_old = data
        data = data[0]
    if isinstance(data, np.void):
        return data_old
    return data


def pascal3d_get_bbox(data):
    names = data.dtype.names
    if 'bbox' in names:
        bbox = data['bbox']
        bbox = get_mat_list(bbox)
        return list(map(float, bbox))
    elif 'bndbox' in names:
        raise Exception("NOT IMPLEMENTED")
    raise PascalParseError("could not parse bounding box")


class PascalParseError(Exception):
    def __init__(self, string):
        super().__init__(string)


class PascalClasses(IntEnum):
    AEROPLANE=1
    BICYCLE=2
    BOAT=3
    BOTTLE=4
    BUS=5
    CAR=6
    CHAIR=7
    DININGTABLE=8
    MOTORBIKE=9
    SOFA=10
    TRAIN=11
    TVMONITOR=12

    def __str__(self):
        return self.name.lower()


pascal_3d_str_enum_map = {}
for v in PascalClasses:
    pascal_3d_str_enum_map[str(v)] = v

failed_parse_strings = set()
def pascal3d_get_class(data):
    class_str = get_mat_element(data['class'])
    try:
        return pascal_3d_str_enum_map[class_str.lower()]
    except KeyError:
        failed_parse_strings.add(class_str.lower())
        # print("unknown class: " + class_str)
        raise PascalParseError("could not parse class")


def pascal3d_idx_to_str(idx):
    return str(PascalClasses(idx))


def parse_single_angle(viewpoint, angle_name):
    names = viewpoint.dtype.names
    if angle_name in names:
        try:
            angle = get_mat_element(viewpoint[angle_name])
            return float(angle)
        except PascalParseError:
            pass
    angle_name_coarse = angle_name + "_coarse"
    if angle_name_coarse in names:
        angle = get_mat_element(viewpoint[angle_name_coarse])
        return float(angle)
    raise PascalParseError("No angle found")


def pascal3d_get_angle(data):
    viewpoint = get_mat_element(data['viewpoint'])
    azimuth = parse_single_angle(viewpoint, 'azimuth')
    elevation = parse_single_angle(viewpoint, 'elevation')
    theta = parse_single_angle(viewpoint, 'theta')
    if azimuth == 0 and elevation == 0 and theta == 0:
        raise PascalParseError("Angle probably not entered")
    return [azimuth, elevation, theta] #note in degree


def pascal3d_get_point(data):
    viewpoint = get_mat_element(data['viewpoint'])
    px = float(get_mat_element(viewpoint['px']))
    py = float(get_mat_element(viewpoint['py']))
    return [px, py]

def pascal3d_get_distance(data):
    viewpoint = get_mat_element(data['viewpoint'])
    return float(get_mat_element(viewpoint['distance']))


DICT_BOUNDING_BOX = 'bounding_box'
DICT_CLASS = 'class'
DICT_ANGLE = 'angle'
DICT_OCCLUDED = 'occluded'
DICT_TRUNCATED = 'truncated'
DICT_DIFFICULT = 'difficult'
DICT_POINT = 'px'
DICT_OBJECT_LIST = 'obj_list'
DICT_OBJECT_INSTANCE = 'obj_instance'
DICT_FILENAME = 'filename'
DICT_DISTANCE = 'distance'
DICT_CAMERA = 'camera'
DICT_CAD_INDEX = 'cad_index'

def get_pascal_camera_params(mat_data):
    viewpoint = get_mat_element(mat_data['viewpoint'])
    try:
        focal = get_mat_element(viewpoint['focal'])
    except PascalParseError:
        print("default_focal")
        focal = 1
    if focal != 1:
        print("focal {}".format(focal))
    try:
        viewport = get_mat_element(viewpoint['viewport'])
    except PascalParseError:
        print("default_viewpoer")
        viewport = 3000
    if viewport != 3000:
        print("viewport {}".format(viewport))
    return float(focal), float(viewport)


def mat_data_to_dict_data(mat_data, folder):
    record = get_mat_element(mat_data['record'])
    ret = {}
    objects = []
    mat_objects = get_mat_list(record['objects'])
    for obj in mat_objects:
        ret_obj = {}
        try:
            ret_obj[DICT_BOUNDING_BOX] = pascal3d_get_bbox(obj)
            ret_obj[DICT_CLASS] = pascal3d_get_class(obj).value
            ret_obj[DICT_ANGLE] = pascal3d_get_angle(obj)
            ret_obj[DICT_OCCLUDED] = bool(get_mat_element(obj['occluded']))
            ret_obj[DICT_TRUNCATED] = bool(get_mat_element(obj['truncated']))
            ret_obj[DICT_POINT] = pascal3d_get_point(obj)
            ret_obj[DICT_DIFFICULT] = bool(get_mat_element(obj['difficult']))
            ret_obj[DICT_DISTANCE] = pascal3d_get_distance(obj)
            ret_obj[DICT_CAMERA] = get_pascal_camera_params(obj)

            ret_obj[DICT_CAD_INDEX] = int(get_mat_element(obj['cad_index']))
            objects.append(ret_obj)
        except PascalParseError as e:
            pass
    ret[DICT_OBJECT_LIST] = objects
    ret[DICT_FILENAME] = os.path.join(folder, get_mat_element(record['filename']))
    return ret

def create_json_annotations(location, json_annotation_path):
    failed_parses = 0
    total_files = 0
    total_parses = 0
    annotation_path = os.path.join(location, 'Annotations')
    os.makedirs(json_annotation_path)
    for folder in os.listdir(annotation_path):
        folder_json_dir = os.path.join(json_annotation_path, folder)
        folder_dir = os.path.join(annotation_path, folder)
        os.makedirs(folder_json_dir)
        for filename in os.listdir(folder_dir):
            total_files += 1
            file_in = os.path.join(folder_dir, filename)
            r, ext = os.path.splitext(filename)
            file_out = os.path.join(folder_json_dir, r + '.json')
            mat_data = scipy.io.loadmat(file_in)
            dict_data = mat_data_to_dict_data(mat_data, folder)
            total_parses += len(dict_data[DICT_OBJECT_LIST])
            if len(dict_data[DICT_OBJECT_LIST]) == 0:
                # print(folder_dir + '/' + filename)
                failed_parses += 1
            if len(dict_data[DICT_OBJECT_LIST]) > 1:
                pass
                # print(filename)
                # print("identified several objects")
            with open(file_out, 'w') as f:
                json.dump(dict_data, f)
    print("fail rate")
    print((failed_parses + 0.0) / total_files)
    print(total_files)
    print(total_parses)

def split_json_annotations(directory_unsplit, directory_split):
    # The point of this function is that there might have been multiple valid objects per image.
    # This function removes the folder structure and 
    os.makedirs(directory_split)
    count = 0
    for folder in os.listdir(directory_unsplit):
        folder_unsplit_dir = os.path.join(directory_unsplit, folder)
        for filename in os.listdir(folder_unsplit_dir):
            load_file = os.path.join(folder_unsplit_dir, filename)
            data = None
            with open(load_file, 'r') as f:
                data = json.load(f)
            for obj in data[DICT_OBJECT_LIST]:
                data_save = copy.deepcopy(data)
                del data_save[DICT_OBJECT_LIST]
                data_save[DICT_OBJECT_INSTANCE] = obj
                save_path = os.path.join(directory_split, '{}.json'.format(count))
                with open(save_path, 'w') as f:
                    json.dump(data_save, f)
                count += 1


def create_split(desired_set, directory_split):
    # The point of this function is to return a list of sample indexes corresponding to a list
    # of folder/imagename entries.
    desired_set = set(desired_set)
    indices = []
    for filename in os.listdir(directory_split):
        full_filename = os.path.join(directory_split, filename)
        with open(full_filename, 'r') as f:
            data = json.load(f)
        img_path = data[DICT_FILENAME]
        sample_path, _ = os.path.splitext(img_path)
        if sample_path in desired_set:
            count, _ = os.path.splitext(filename)
            indices.append(count)
    return indices


def RQ(A):
    to_qr = A.transpose()[:, ::-1]
    Qhat, Rhat = np.linalg.qr(to_qr)
    return (Rhat[::-1, ::-1].transpose()), Qhat[:, ::-1].transpose()

def split_camera_matrix(P):
    extrinsic = np.zeros((4,4))
    intrinsic = np.zeros((3,3))
    extrinsic[3,3] = 1
    r,q  = RQ(P[:3,:3])
    for i in range(3):
        if r[i,i] < 0:
            q[i, :] *= -1
            r[:, i] *= -1

    assert(np.linalg.det(q) > 0)
    extrinsic[:3,:3] = q
    intrinsic = r
    intrinsic /= intrinsic[2,2]
    extrinsic[:3,3] = np.linalg.solve(intrinsic, P[:3, 3])

    if np.max((np.matmul(intrinsic, extrinsic[:3, :]) - P).reshape(-1)) > 0.1:
        print(P)
        print(np.matmul(intrinsic, extrinsic[:3, :]))
        raise Exception("Failed to split camera")
    return intrinsic, extrinsic


def get_camera_matrix(camera, principal_point, angle, image_size, distance):
    f = np.prod(camera)
    a = angle[0] *np.pi/180
    e = angle[1] *np.pi/180
    th = angle[2] *np.pi/180
    C = np.zeros(3)

    C[0] = distance*np.cos(e)*np.sin(a)
    C[1] = -distance*np.cos(e)*np.cos(a)
    C[2] = distance*np.sin(e)

    a = -a
    e = e-np.pi/2

    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0,np.sin(e), np.cos(e)]])
    R = np.matmul(Rx, Rz)

    R2d = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0,0,1]])
    intrinsic = np.array([[f, 0, 0], [0, f, 0], [0,0,-1]])
    intrinsic_w_rot = np.matmul(R2d, intrinsic)
    y_flip = np.array([[1, 0,0], [0,-1,0], [0,0,1]])
    intrinsic_w_flip = np.matmul(y_flip, intrinsic_w_rot)
    pp = principal_point # + np.array([image_size[1]/2.0, image_size[0]/2.0]) # image size is hxw
    trans_mat = np.array([[1.0, 0, pp[0]], [0, 1, pp[1]], [0,0,1]])
    intrinsic_w_trans = np.matmul(trans_mat, intrinsic_w_flip)

    extrinsic = np.zeros((4,4))
    extrinsic[:3,:3] = R
    extrinsic[3,3] = 1
    extrinsic[:3,3] = -np.matmul(R, C)
    P = np.matmul(intrinsic_w_trans, extrinsic[:3,:])
    return P


def get_camera_parameters(cam, principal_point, angle, image_size, distance):
    P = get_camera_matrix(cam, principal_point, angle, image_size, distance)
    intrinsic, extrinsic = split_camera_matrix(P)
    return intrinsic, extrinsic


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


def get_desired_camera(desired_imagesize, backproj, desired_up):
    z, radius_3d = get_minimum_covering_sphere(backproj.transpose())
    y = desired_up - np.dot(desired_up, z)*z
    y /= np.linalg.norm(y)
    x = -np.cross(y,z)
    R = np.stack([y,x,z], axis=0)# AXIS? TODO
    bp_reproj = np.matmul(R, backproj)
    bp_reproj/=bp_reproj[2,:].reshape(1, -1)
    f = 1/np.max(np.abs(bp_reproj[:2]).reshape(-1))

    intrinsic = np.array([[desired_imagesize*f/2, 0, desired_imagesize/2],
                          [0, desired_imagesize*f/2, desired_imagesize/2],
                          [0, 0, 1]])
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = R
    return extrinsic, intrinsic


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


def show_crop_bounding_box(img, bbox):
    shape = img.shape
    minx = int(round(bbox[0]))
    miny = int(round(bbox[1]))
    maxx = int(round(bbox[2]))
    maxy = int(round(bbox[3]))
    sz_x = maxx-minx+1
    sz_y = maxy-miny+1
    im_show = np.zeros((sz_y, sz_x, 3))
    if minx < 0:
        offset_x = -minx
        full_start_x = 0
    else:
        offset_x = 0
        full_start_x = minx
    if miny < 0:
        offset_y = -miny
        full_start_y = 0
    else:
        offset_y = 0
        full_start_y = miny
    if maxx < shape[1]:
        end_x = sz_x
        full_end_x = maxx
    else:
        end_x = shape[1]-minx
        full_end_x = shape[1]
    if maxy < shape[0]:
        end_y = sz_y
        full_end_y = maxy
    else:
        end_y = shape[0]-miny
        full_end_y = shape[0]
    im_show[offset_y:end_y, offset_x:end_x] = img[full_start_y:full_end_y+1, full_start_x:full_end_x+1] / 255
    plt.imshow(im_show)
    plt.show()




import shutil
class Pascal3D():
    def __init__(self, dataset_location=None, image_size=224, train_all=False, use_warp=True, voc_train=False):
        print('start init pascal')
        self.image_out_size = image_size
        self.voc_train = voc_train
        if dataset_location==None:
            dataset_location = '/home/tobii.intra/dmon/datasets'
        self.location = os.path.join(dataset_location, DATASET_FOLDER_NAME)
        if not os.path.exists(self.location):
            raise Exception("Not implemented yet")
            download_pascal3d(self.location) #TODO unzip as well
        if not os.path.exists(self.json_annotation_path()):
            json_annotation_path_tmp = os.path.join(self.location, 'json_tmp')
            if os.path.exists(json_annotation_path_tmp):
                shutil.rmtree(json_annotation_path_tmp)
            create_json_annotations(self.location, json_annotation_path_tmp)
            split_json_annotations(json_annotation_path_tmp, self.json_annotation_path())
        train_idx, val_idx, test_idx = self.get_split(0.3)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.train_all = train_all
        self.use_warp = use_warp

    def json_annotation_path(self):
        return os.path.join(self.location, JSON_ANNOTATION_FOLDER_NAME)

    def dataset_split_path(self):
        return os.path.join(self.location, 'Image_sets')

    def image_set_path(self):
        return os.path.join(self.location, 'Images')

    def stored_split_index(self):
        if self.voc_train:
            split_name = 'saved_splits_pascal.txt'
        else:
            split_name = 'saved_splits.txt'
        return os.path.join(self.location, split_name)

    def get_split(self, validation_split_size=0.3):
        stored_indices_path = self.stored_split_index()
        if os.path.exists(stored_indices_path):
            with open(stored_indices_path, 'r') as f:
                return load_split(f)
        split_names = ['train', 'val']
        splits = []
        num_files = 0
        dataset_split_path = self.dataset_split_path()
        for split_name in split_names:
            split = []
            for cls in PascalClasses:
                filepath = os.path.join(dataset_split_path, str(cls) + '_imagenet_' + split_name + '.txt')
                with open(filepath, 'r') as f:
                    while True:
                        l = f.readline()
                        if len(l) == 0:
                            break
                        while l[-1] in ('\n', '\r'):
                            l = l[:-1]
                            if len(l) == 0:
                                continue
                        num_files += 1
                        img_path = os.path.join(str(cls) + '_imagenet', l + '.JPEG')
                        split.append(img_path)
            splits.append(split)
        split_pascal = []
        for directory in os.listdir(self.image_set_path()):
            if directory.lower().find('pascal') != -1:
                for fname in os.listdir(os.path.join(self.image_set_path(), directory)):
                    split_pascal.append(os.path.join(directory, fname))
        test_split = splits[1]
        rest = splits[0]
        rest = sorted(rest)
        val_idx = (np.arange(len(rest)*validation_split_size)/validation_split_size).astype(np.int)
        val_split = [rest[i] for i in val_idx]
        train_split = sorted(list(set(rest)-set(val_split)))
        if self.voc_train:
            train_split = train_split + split_pascal

        ret = get_dataset_indices_from_imagefilenames([train_split, val_split, test_split], self.json_annotation_path())
        with open(stored_indices_path, 'w') as f:
            save_split(f, ret)
        return ret


    def get_train(self, augmentation=False):
        if self.train_all:
            return self.get_subset(self.train_idx + self.val_idx, augmentation)
        else:
            return self.get_subset(self.train_idx, augmentation)

    def get_eval(self):
        if self.train_all:
            return self.get_subset(self.test_idx, False)
        else:
            return self.get_subset(self.val_idx, False)

    def get_cad(self, class_idx, cad_idx):
        class_str = pascal3d_idx_to_str(class_idx)
        cad_file = os.path.join(self.location, 'CAD', class_str, '{0:02d}.off'.format(cad_idx))
        with open(cad_file, 'r') as f:
            vertices, faces = parse_off_file(f)
        return np.array(vertices), np.array(faces)

    def get_subset(self, index, augmentation):
        if self.use_warp:
            return Pascal3DSubsetWarp(self.json_annotation_path(), self.image_set_path(), index, self.image_out_size, augmentation)
        else:
            assert(augmentation is False)
            return Pascal3DSubsetCrop(self.json_annotation_path(), self.image_set_path(), index, self.image_out_size)

class Pascal3DSubsetWarp(Dataset):
    def __init__(self, json_directory, image_directory, files, desired_image_out, augmentation):
        self.json_directory = json_directory
        self.image_directory = image_directory
        self.files = files
        self.image_out_size = desired_image_out
        self.augmentation = augmentation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_file = os.path.join(self.json_directory, "{}.json".format(self.files[idx]))
        sample_data = None
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
        im_path = os.path.join(self.image_directory, sample_data[DICT_FILENAME])
        with open(im_path, 'rb') as f:
            img_PIL = Image.open(f)
            img_PIL.convert('RGB')
            data = img_PIL.getdata()
            if isinstance(data[0], np.int) or len(data[0]) == img_PIL.size[1] * img_PIL.size[0]:
                img_full = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0]).reshape(img_PIL.size[1], img_PIL.size[0],1).repeat(3,2)
            else:
                img_full = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)
        obj = sample_data[DICT_OBJECT_INSTANCE]
        bbox = obj[DICT_BOUNDING_BOX]
        class_idx = obj[DICT_CLASS]
        angle = obj[DICT_ANGLE]
        cam = np.array(obj[DICT_CAMERA])
        principal_point = np.array(obj[DICT_POINT])
        distance = obj[DICT_DISTANCE]

        if self.augmentation:
            flip = np.random.randint(2)
            if flip==1:
                angle *= np.array([-1.0, 1.0, -1.0])
                img_full = img_full[:, ::-1, :]
                bbox[0] = img_full.shape[1] - bbox[0]
                bbox[2] = img_full.shape[1] - bbox[2]
                principal_point[0] = img_full.shape[1] - principal_point[0]
            # # change up direction of warp
            desired_up = np.random.normal(0,0.4,size=(3))+np.array([3.0, 0.0, 0.0])
            desired_up[2] = 0
            desired_up /= np.linalg.norm(desired_up)
            # # jitter bounding box
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox[0::2] += np.random.uniform(-bbox_w*0.1, bbox_w*0.1, size=(2))
            bbox[1::2] += np.random.uniform(-bbox_h*0.1, bbox_h*0.1, size=(2))
        else:
            desired_up = np.array([1.0, 0.0, 0.0])

        intrinsic, extrinsic = get_camera_parameters(cam, principal_point, angle, img_full.shape, distance)
        back_proj_bbx = get_back_proj_bbx(bbox, intrinsic)

        desired_imagesize=self.image_out_size
        extrinsic_desired_change, intrinsic_new = get_desired_camera(desired_imagesize, back_proj_bbx, desired_up)
        extrinsic_after = np.matmul(extrinsic_desired_change, extrinsic)

        P = np.matmul(np.matmul(intrinsic_new, extrinsic_desired_change[:3, :3]), np.linalg.inv(intrinsic))
        P /= P[2,2]
        Pinv = np.linalg.inv(P)
        transform = skimage.transform.ProjectiveTransform(Pinv)
        im = img_full.astype(np.float)/255
        warped_image = skimage.transform.warp(im, transform, output_shape=(desired_imagesize, desired_imagesize), mode='constant', cval=0.0)
        occluded = obj[DICT_OCCLUDED]
        truncated = obj[DICT_TRUNCATED]
        difficult = obj[DICT_DIFFICULT]
        cad_idx = obj[DICT_CAD_INDEX]
        hard = occluded or truncated or difficult
        return warped_image.transpose(2,0,1).astype(np.float32), extrinsic_after.astype(np.float32), int(class_idx), hard, intrinsic_new.astype(np.float32), int(cad_idx)


class Pascal3DSubsetCrop(Dataset):
    def __init__(self, json_directory, image_directory, files, desired_image_out):
        self.json_directory = json_directory
        self.image_directory = image_directory
        self.files = files
        self.image_out_size = desired_image_out

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_file = os.path.join(self.json_directory, "{}.json".format(self.files[idx]))
        sample_data = None
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
        im_path = os.path.join(self.image_directory, sample_data[DICT_FILENAME])
        with open(im_path, 'rb') as f:
            img_PIL = Image.open(f)
            img_PIL.convert('RGB')
            data = img_PIL.getdata()
            if isinstance(data[0], np.int) or len(data[0]) == img_PIL.size[1] * img_PIL.size[0]:
                img_full = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0]).reshape(img_PIL.size[1], img_PIL.size[0],1).repeat(3,2)
            else:
                img_full = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)
        obj = sample_data[DICT_OBJECT_INSTANCE]
        bbox = obj[DICT_BOUNDING_BOX]
        class_idx = obj[DICT_CLASS]
        angle = obj[DICT_ANGLE]
        cam = np.array(obj[DICT_CAMERA])
        principal_point = np.array(obj[DICT_POINT])
        distance = obj[DICT_DISTANCE]

        intrinsic, extrinsic = get_camera_parameters(cam, principal_point, angle, img_full.shape, distance)
        back_proj_bbx = get_back_proj_bbx(bbox, intrinsic)

        bbox[0] = int(round(min(img_full.shape[1]-1, max(0, bbox[0]))))
        bbox[1] = int(round(min(img_full.shape[0]-1, max(0, bbox[1]))))
        bbox[2] = int(round(min(img_full.shape[1], max(0, bbox[2]))))
        bbox[3] = int(round(min(img_full.shape[0], max(0, bbox[3]))))
        im = img_full[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # plt.imshow(img_full)
        # x = [bbox[0], bbox[0], bbox[2], bbox[2], bbox[0]]
        # y = [bbox[1], bbox[3], bbox[3], bbox[1], bbox[1]]
        # plt.plot(x,y,'r', linewidth=5)
        # plt.show()
        im = np.array(Image.fromarray(im.astype(np.uint8)).resize((self.image_out_size, self.image_out_size), Image.BILINEAR))
        im = im.astype(np.float32)/255
        intrinsic = np.array([[self.image_out_size*8, 0, self.image_out_size/2],
                              [0, self.image_out_size*8, self.image_out_size/2],
                              [0, 0, 1]])


        occluded = obj[DICT_OCCLUDED]
        truncated = obj[DICT_TRUNCATED]
        difficult = obj[DICT_DIFFICULT]
        cad_idx = obj[DICT_CAD_INDEX]
        hard = occluded or truncated or difficult
        return im.transpose(2,0,1).astype(np.float32), extrinsic.astype(np.float32), int(class_idx), hard, intrinsic.astype(np.float32), int(cad_idx)




def get_dataset_indices_from_imagefilenames(splits, json_annotation_path):
    for i in range(len(splits)):
        for j in range(len(splits)):
            if i != j:
                intersect = set(splits[i]).intersection(set(splits[j]))
                if len(intersect) != 0:
                    print('split {} and {} have filename intersection {}'.format(i, j, intersect))
                    raise Exception('train/val/test filename split intersects')

    resmap = {}
    for idx in range(len(os.listdir(json_annotation_path))):
        sample_file = os.path.join(json_annotation_path, "{}.json".format(idx))
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
        filename = sample_data[DICT_FILENAME]
        idx_for_file = resmap.get(filename, [])
        idx_for_file.append(idx)
        resmap[filename] = idx_for_file
    ret = []
    for split in splits:
        split_ret = []
        for filename in split:
            to_add = resmap.get(filename, [])
            split_ret += to_add
        ret.append(sorted(split_ret))
    for i in range(len(ret)):
        for j in range(len(ret)):
            if i != j:
                intersect = set(ret[i]).intersection(set(ret[j]))
                if len(intersect) != 0:
                    print('split {} and {} have index intersection {}'.format(i, j, intersect))
                    raise Exception('train/val/test index split intersects')
    return ret

def save_split(f, splits):
    for split in splits:
        print(' '.join(map(str, split)), file=f)

def load_split(f):
    l = f.readline()
    ret = []
    while len(l) != 0:
        ret.append(list(map(int, l.split(' '))))
        l = f.readline()
    return ret
