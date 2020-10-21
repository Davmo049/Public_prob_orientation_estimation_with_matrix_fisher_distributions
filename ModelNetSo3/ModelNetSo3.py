import lmdb
import os
import io
import pickle
from PIL import Image
from geometric_utils import numpy_euler_to_rotation, numpy_quaternion_to_rot_mat, numpy_aet_to_rot
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum

dataset_dir = 'dataset' # change to where you store datasets

test_name = 'test_20V.Rawjpg.lmdb'
train_name = 'train_100V.Rawjpg.lmdb'

class ModelNetSo3():
    def __init__(self, path=dataset_dir):
        self.modelnet_dir = os.path.join(path, 'ModelNet10-SO3')

    def get_train(self):
        full = ModelNetSo3Subset(os.path.join(self.modelnet_dir, train_name)) # split TODO
        return full

    def get_eval(self):
        return ModelNetSo3Subset(os.path.join(self.modelnet_dir, test_name))

    def get_cad(self, class_idx, cad_idx):
        return np.array([[0.0,0,0], [1,0,0], [0,1,0], [0,0,1]]), None

class ModelNetSo3Classes(IntEnum):
    BATHTUB=0
    BED=1
    CHAIR=2
    DESK=3
    DRESSER=4
    MONITOR=5
    NIGHT_STAND=6
    SOFA=7
    TABLE=8
    TOILET=9

    def __str__(self):
        return self.name.lower()


modelnetso3_str_enum_map = {}
for v in ModelNetSo3Classes:
    modelnetso3_str_enum_map[str(v)] = v

class ModelNetSo3Subset():
    def __init__(self, path, imsize=224):
        self.path = path
        lmdb_db = lmdb.open(path)
        quaternions_path = os.path.join(path, 'viewID2quat.pkl')
        euler_path = os.path.join(path, 'viewID2euler.pkl')
        with open(quaternions_path, 'rb') as f:
            quaterions = pickle.load(f, encoding='latin1')
        with open(euler_path, 'rb') as f:
            euler = pickle.load(f, encoding='latin1')
        if os.path.exists(self.key_path()):
            with open(self.key_path(), 'r') as f:
                keys = [x.strip() for x in f.readlines()]
        else:
            with lmdb_db.begin() as txn:
                keys = [key.decode('utf-8') for key, _ in txn.cursor()]
            with open(self.key_path(), 'w') as f:
                f.write('\n'.join(keys))

        self.imsize = imsize
        self.lmdb = lmdb_db
        self.keys = keys
        self.quaterions = quaterions
        self.euler = euler

    def key_path(self):
        return os.path.join(self.path, 'keys.txt')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        cad_id, view_id = key.split('.')
        class_str = cad_id[:cad_id.rfind('_')]
        
        class_idx = modelnetso3_str_enum_map[class_str]
        with self.lmdb.begin() as txn:
            bits = txn.get(key.encode('utf8'))
        f = io.BytesIO(bits)
        image = Image.open(f)
        image = image.resize((self.imsize, self.imsize), Image.ANTIALIAS)
        image = np.array(image.getdata()).reshape(self.imsize, self.imsize, 1).repeat(3,2)
        image = image.astype(np.float32) / 255
        image = image.transpose(2,0,1)
        q = self.quaterions[key]
        qx = np.array([q[1], -q[2], q[3], q[0]])
        R = numpy_quaternion_to_rot_mat(qx)
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = R
        extrinsic[:3,3] = np.array([0,0,5])
        intrinsic = np.array([[self.imsize/2, 0, self.imsize/2],
                              [0, self.imsize/2, self.imsize/2],
                              [0,0,1]])
        extrinsic=extrinsic.astype(np.float32)
        return image, extrinsic, class_idx, 0, intrinsic, 0




def main():
    dataset = ModelNetSo3()
    tr = dataset.get_train()
    print(len(tr))
    exit(0)
    for i in range(50000):
        image, extrinsic, class_idx, hard, intrinsic, cad_idx = tr[i]
        nodes, _ = dataset.get_cad(class_idx, cad_idx)
        nodes_homo = np.ones((4, len(nodes)))
        nodes_homo[:3, :] = nodes.transpose()
        model = np.matmul(extrinsic, nodes_homo)
        model = model[:3, :]
        model /= model[2].reshape(1, -1)
        mod_proj = np.matmul(intrinsic, model)
        plt.imshow(image.transpose(1,2,0))
        for i,c in enumerate(['r','g','b']):
            x = [mod_proj[0,0], mod_proj[0, i+1]]
            y = [mod_proj[1,0], mod_proj[1, i+1]]
            plt.plot(x, y, c)
        plt.show()

if __name__ == '__main__':
    main()
