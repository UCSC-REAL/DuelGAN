# DuelGAN: A Duel Between Two Discriminators Stabilizes the GAN Training
ECCV 2022 [Paper](https://arxiv.org/abs/2101.07524)

**Jiaheng Wei***, **Minghao Liu***, Jiahao Luo, Andrew Zhu, James Davis, Yang Liu

Co-First *



## Getting Started

### Installation
- Clone this repo:

- Install the dependencies
```bash
conda create --name dualgan python=3.6
conda activate dualgan
conda install --file requirements.txt
conda install -c conda-forge tensorboardx
```
### Training and Evaluation
- Train a model on CIFAR:
```bash
python train.py configs/dual/cifar10.yaml
```


- Evaluate the model's FID:
You will need to first gather a set of ground truth train set images to compute metrics against.
```bash
python utils/get_gt_imgs.py --cifar
python metrics.py configs/dual/cifar10.yaml --fid --every -1
```

## Using Dual Game in your GAN model
We used two identical discriminator in the pipeline. 
```python
import torch
import torch.nn as nn
import numpy as np


def _dual_loss(d_to_backprop, d_dual, x_real, x_fake, alpha=0.3,detach_g=True):
    dual_losses = []
    x1, x2, x3 = choose_random_samples((3, x_real.shape[0]), x_real, x_fake, detach_g)

    # It could be your choose of adversarial loss such as CE. 
    # Or use the stable version of adversarial loss from us.
    p1 = adversarial_loss(d_to_backprop(x1), d_dual(x1).detach())
    p2 = adversarial_loss(d_to_backprop(x2), d_dual(x3).detach())
    l = p1 - alpha * p2
    dual_losses.append(l)

    dual_loss = torch.mean(torch.stack(dual_losses))
    return dual_loss, p1, p2


def choose_random_samples(self, shape, x_real, x_fake, detach_g):

    samples = []
    num_samples, batch_size = shape

    for i in range(num_samples):

        np.random.shuffle(x_real)
        np.random.shuffle(x_fake)

        half = int(batch_size / 2)
        if detach_g:
            x_real_half = x_real[:half, :, :, :].detach()
            x_fake_half = x_fake[:half, :, :, :].detach()
        else:
            x_real_half = x_real[:half, :, :, :]
            x_fake_half = x_fake[:half, :, :, :]

        out = torch.cat((x_real_half, x_fake_half), 0)
        np.random.shuffle(out)
        samples.append(out)

    return tuple(samples)
```
## Adversarial loss

```python
# Don't forget to initialize the Adversarial loss

# in your init
adversarial_loss_stable = BCE_Stable()

def adversarial_loss(d_out, target):
    loss = adversarial_loss_stable(torch.sigmoid(d_out), torch.sigmoid(target))
    return loss

# Class of stable BCE loss
class BCE_Stable(nn.Module):
    '''
    To avoid blowup when using in the dual term
    '''

    def __init__(self, reduction='mean', eps=1e-8):
        super(BCE_Stable, self).__init__()
        self._eps = eps
        self._sigmoid = nn.Sigmoid()
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        log_out = torch.log(self._sigmoid(outputs) + self._eps)
        res = torch.sum(torch.mul(labels, log_out), dim=0)

        return -torch.mean(res)
```

## Acknowledgments
This code is heavily based on the [Diverse Image Generation via Self-Conditioned GANs](https://github.com/stevliu/self-conditioned-gan) code base.

To compute FID, we use the code provided from [TTUR](https://github.com/bioinf-jku/TTUR).

We thank all the authors for their useful code.
