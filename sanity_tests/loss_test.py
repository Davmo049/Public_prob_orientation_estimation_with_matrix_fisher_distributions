import loss
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import geometric_utils
from geometric_utils import numpy_quaternion_to_rot_mat, numpy_quaternion_to_angles, numpy_euler_to_rotation
from batch_norm_utils import bn_set_running_stats, bn_set_global_run_stats

def dataset_generator(N, shuffle_points=False, positive_w_quat=False, quaternion_generation=True):
    r = []
    x = np.array([[1.0, 0.0,0.0], [0.0, 1.0,0.0], [0.0, 0.0,1.0]]).transpose()
    for n in range(N):
        if quaternion_generation:
            q = np.random.normal(size=(4))
            R = numpy_quaternion_to_rot_mat(q)
            angles = numpy_quaternion_to_angles(q)
        else:
            angles = np.random.uniform(-np.pi/4,np.pi/4, (3))
            R = numpy_euler_to_rotation(angles)
            q = np.random.normal(size=(4)) # not correct, fix TODO
        xr = np.matmul(R,x).transpose()
        if shuffle_points:
            np.random.shuffle(xr)
        if positive_w_quat:
            q *= np.sign(q[3])
        feature = xr.flatten()
        r.append((feature, R, q, angles))
    return r


def torch_generator(bs=32):
    q = torch.normal(0, 1, size=(bs, 4))
    q /= torch.norm(q, dim=1).view(-1, 1)
    R = geometric_utils.torch_batch_quaternion_to_rot_mat(q)
    return [None, R, q, None]


def torch_batch_dataset(dataset, bs=32):
    batches = []
    for idx in range(len(dataset) // bs):
        idx_start = idx*bs
        idx_end = (idx+1)*bs
        x_batch = []
        r_batch = []
        q_batch = []
        a_batch = []
        for i in range(idx_start, idx_end):
            x,r,q,a = dataset[i]
            x_batch.append(x)
            r_batch.append(r)
            q_batch.append(q)
            a_batch.append(a)
        batches.append((torch.tensor(x_batch, dtype=torch.float32), torch.tensor(r_batch, dtype=torch.float32), torch.tensor(q_batch, dtype=torch.float32), torch.tensor(a_batch, dtype=torch.float32)))
    return batches

def get_ann(n_in=9, n_outputs=9, n_hidden_nodes=32, use_bn=False):
    if use_bn:
        model = torch.nn.Sequential(*[
            torch.nn.Linear(n_in, n_hidden_nodes),
            torch.nn.BatchNorm1d(n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.BatchNorm1d(n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.BatchNorm1d(n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.BatchNorm1d(n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_outputs)
        ])
    else:
        model = torch.nn.Sequential(*[
            torch.nn.Linear(n_in, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_outputs)
        ])

    return model

def train_direct_quat_L2(dtrain, dtest, epochs=5):
    print('direct_quat')
    model = get_ann(9, 4)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = epochs
    for epoch in range(num_epochs):
        print('epoch: {}'.format(epoch))
        ls = 0.0
        angerr = 0.0
        for batch in dtrain:
            out = model(batch[0])
            lossall, Rest = loss.direct_quat_loss(out, batch[2])
            loss_v = torch.mean(lossall)
            optim.zero_grad()
            loss_v.backward()
            optim.step()
            a_err = loss.angle_error(Rest, batch[1])
            angerr += torch.mean(a_err).detach().numpy()
            ls += loss_v.detach().numpy()
        print('traloss')
        print(ls / batches_per_epoch)
        print(angerr / batches_per_epoch)
        ls = 0.0
        angerr = []
        with torch.no_grad():
            for batch in dtest:
                model.eval()
                out = model(batch[0])
                lossall, Rest = loss.direct_quat_loss(out, batch[2])
                loss_v = torch.mean(lossall)
                a_err = loss.angle_error(Rest, batch[1])
                ls += loss_v.detach().numpy()
                angerr += list(a_err.detach().numpy())
        print('testloss')
        print(np.mean(angerr))
        print(ls / len(dtest))
        if epoch == 0 or epoch == num_epochs-1:
            plt.hist(angerr, 500)
            print(np.mean(angerr))
            print(np.sum(np.array(angerr) > 120))
            plt.show()



def train_quat_L2(dtrain, dtest, epochs=5):
    print('quat_with_R')
    model = get_ann(9, 4)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = epochs
    for epoch in range(num_epochs):
        print('epoch: {}'.format(epoch))
        ls = 0.0
        angerr = 0.0
        model.train()
        for batch in dtrain:

            out = model(batch[0])
            lossall, Rest = loss.quat_R_loss(out, batch[1])
            loss_v = torch.mean(lossall)
            optim.zero_grad()
            loss_v.backward()
            optim.step()
            a_err = loss.angle_error(Rest, batch[1])
            angerr += torch.mean(a_err).detach().numpy()
            ls += loss_v.detach().numpy()
        print('traloss')
        print(ls / batches_per_epoch)
        print(angerr / batches_per_epoch)
        ls = 0.0
        angerr = []
        with torch.no_grad():
            for batch in dtest:
                model.eval()
                out = model(batch[0])
                lossall, Rest = loss.quat_R_loss(out, batch[1])
                loss_v = torch.mean(lossall)
                a_err = loss.angle_error(Rest, batch[1])
                ls += loss_v.detach().numpy()
                angerr += list(a_err.detach().numpy())
        print('testloss')
        print(np.mean(angerr))
        print(ls / len(dtest))
        if epoch == 0 or epoch == num_epochs-1:
            plt.hist(angerr, 500)
            print(np.mean(angerr))
            print(np.sum(np.array(angerr) > 120))
            plt.show()


def clamp_gradient_magnitude(grad):
    grad_flat = grad.view(grad.shape[0], -1)
    grad_mag = torch.norm(grad_flat, dim=1).view(-1, 1)
    grad_clamp = torch.clamp(grad_mag, 0, 1/32)
    scale = (grad_clamp/grad_mag)
    ret = (grad_flat * scale).view(grad.shape)
    return ret



def train_vmf(bs, batches_per_epoch, dtest, epochs=5):
    print('vmf')
    model = get_ann(4,9, 256, True)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = epochs
    for epoch in range(num_epochs):
        print('epoch: {}'.format(epoch))
        rerun_batch_norm = epoch % 10 == 0
        if epoch == 9:
            optim = torch.optim.Adam(model.parameters(), lr=0.00001)
        elif epoch == 6:
            optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        elif epoch == 3:
            optim = torch.optim.Adam(model.parameters(), lr=0.001)
        ls = 0.0
        first_batch = True
        angerr = 0.0
        model.train()
        for _ in range(batches_per_epoch):
            batch = torch_generator(bs)
            out = model(batch[2]).view(-1, 3,3)
            loss_v = torch.mean(loss.KL_approx(out, batch[1], dbg=False))
            optim.zero_grad()
            loss_v.backward()
            optim.step()
            R = loss.batch_torch_A_to_R(out)
            a_err = loss.angle_error(R, batch[1])
            ls += loss_v.detach().numpy()
            angerr += torch.mean(a_err).detach().numpy()
            first_batch = False
        print('traloss')
        print(ls / batches_per_epoch)
        print(angerr / batches_per_epoch)
        ls = 0.0

        # if rerun_batch_norm:
        #     bn_set_global_run_stats(model)
        #     for batch in dtrain:
        #         out = model(batch[2]).view(-1, 3,3)
        #         loss_v = torch.mean(loss.KL_approx(out, batch[1], dbg=False))
        #     bn_set_running_stats(model, 0.1)

        angerr = []
        with torch.no_grad():
            model.eval()
            for batch in dtest:
                out = model(batch[2]).view(-1, 3,3)
                loss_v = torch.mean(loss.KL_approx(out, batch[1]))
                R = loss.batch_torch_A_to_R(out)
                a_err = loss.angle_error(R, batch[1])
                ls += loss_v.detach().numpy()
                angerr += list(a_err.detach().numpy())
        print('testloss')
        print(ls / len(dtest))
        print(np.mean(angerr))
        if epoch == 0 or epoch == num_epochs-1:
            plt.hist(angerr, 500)
            print(np.sum(np.array(angerr) > 120))
            plt.show()


def train_euler_L2(dtrain, dtest, epochs=5):
    print('euler_L2')
    model = get_ann(9, 3)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = epochs
    out_scale = 0.1
    for epoch in range(num_epochs):
        print('epoch: {}'.format(epoch))
        ls = 0.0
        first_batch = True
        angerr = 0.0
        for batch in dtrain:
            model.train()
            out = model(batch[0])
            angle = torch.tanh(out_scale*out)*torch.tensor([np.pi, np.pi/2, np.pi]).view(-1,3)
            lossall, Rest = loss.euler_loss(angle, batch[3])
            loss_v = torch.mean(lossall)
            optim.zero_grad()
            loss_v.backward()
            optim.step()
            a_err = loss.angle_error(Rest, batch[1])
            ls += loss_v.detach().numpy()
            angerr += torch.mean(a_err).detach().numpy()
            first_batch = False
        print('traloss')
        print(ls / batches_per_epoch)
        print(angerr / batches_per_epoch)
        ls = 0.0
        angerr = []
        with torch.no_grad():
            for batch in dtest:
                model.eval()
                out = model(batch[0])
                angle = torch.tanh(out_scale*out)*torch.tensor([np.pi, np.pi/2, np.pi]).view(-1,3)
                lossall, Rest = loss.euler_loss(angle, batch[3])
                loss_v = torch.mean(lossall)
                a_err = loss.angle_error(Rest, batch[1])
                ls += loss_v.detach().numpy()
                angerr += list(a_err.detach().numpy())
        print('testloss')
        print(ls / len(dtest))
        print(np.mean(angerr))
        if epoch == 0 or epoch == num_epochs-1:
            plt.hist(angerr, 500)
            print(np.sum(np.array(angerr) > 120))
            plt.show()





def main():
    bs = 128*10
    test_dataset = dataset_generator(bs*100, shuffle_points=False, positive_w_quat=False)
    test_dataset = torch_batch_dataset(test_dataset, bs=bs)
    epochs = 100
    batches_per_epoch = 100
    train_vmf(bs, batches_per_epoch, test_dataset, epochs)
    # train_quat_L2(train_dataset, test_dataset, epochs)
    # train_direct_quat_L2(train_dataset, test_dataset, epochs)
    # train_euler_L2(train_dataset, test_dataset, epochs)



if __name__ == '__main__':
    main()
