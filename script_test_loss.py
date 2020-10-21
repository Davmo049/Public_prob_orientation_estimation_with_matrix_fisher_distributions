import loss
import numpy as np
import torch

def main():
    d = 11
    Ft = torch.tensor([[d, 0, 0], [0,d,0], [0,0,d]], dtype=torch.float32, requires_grad=True)
    F = Ft.view(-1, 3,3)
    R = torch.tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32)
    loss_v = loss.KL_Fisher(F,R, overreg=1.05)
    loss_v = torch.sum(loss_v)
    loss_v.backward()
    print(Ft.grad)
    Ft.grad=None
    F = Ft.view(-1, 3,3)
    R = torch.tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32)
    loss_a = loss.KL_approx_rough(F,R, overreg=1.05)
    loss_a = torch.sum(loss_a)
    loss_a.backward()
    print(Ft.grad)



if __name__ == '__main__':
    main()
