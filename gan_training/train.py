# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from gan_training import utils
from torch.autograd import Variable
import time
import torch.nn as nn


class Trainer(object):
    def __init__(self,
                 generator,
                 discriminator1,
                 discriminator2,
                 g_optimizer,
                 d1_optimizer,
                 d2_optimizer,
                 gan_type,
                 reg_type,
                 reg_param,
                 train_loader,
                 Tensor,
                 dim,
                 alpha,
                 beta):

        self.generator = generator
        self.discriminator1 = discriminator1
        self.discriminator2 = discriminator2
        self.g_optimizer = g_optimizer
        self.d1_optimizer = d1_optimizer
        self.d2_optimizer = d2_optimizer
        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.train_loader = (train_loader)
        self.iter_train_loader = iter(self.train_loader)
        self.train_loader_dataset = train_loader.dataset
        self.n_samples = len(self.train_loader_dataset)
        self.Tensor = Tensor
        self.dim = dim
        self.alpha = alpha
        self.beta = beta

        self.adversarial_loss_stable = BCE_Stable()
        self.criterion_log = Log_loss()
        self.criterion_itself = Itself_loss()

        print('D reg gamma', self.reg_param)

    def update_dual_param(self, curr_epoch, all_epoch, param_type,config):
        max_value_alpha = config['dual_alpha'] #0.3
        max_value_beta = config['dual_beta'] #0.5
        # max_value_gamma = 0.5

        ratio = curr_epoch / all_epoch
        x = ratio * 100
        y_alpha = -0.0004 * x * (x - 100) * max_value_alpha
        y_beta = -0.0004 * x * (x - 100) * max_value_beta
        # y_gamma = -0.0004 * x * (x - 100) * max_value_gamma

        # update values  
        if param_type == 'alpha':
            self.alpha = y
        elif param_type == 'beta':
            self.beta = y
        elif param_type == 'all':
            self.alpha = y_alpha
            self.beta = y_beta
        elif param_type == 'fix':
            self.alpha = self.alpha
            self.beta = self.beta

        return self.alpha, self.beta

    def generator_trainstep(self, x_real, z):
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator1, False)
        toggle_grad(self.discriminator2, False)

        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z)

        #############
        # balanced
        #############
        x_real.requires_grad_()
        x_fake.requires_grad_()
        d1_fake = self.discriminator1(x_fake)
        d2_fake = self.discriminator2(x_fake)

        g_loss = (self.compute_loss(d1_fake, 1) + self.compute_loss(d2_fake, 1)) / 2
        # g_loss = self.compute_loss(d1_fake, 1)
        g_loss.backward()

        self.g_optimizer.step()
        return g_loss.item()

    def discriminator1_trainstep(self, x_real, z):

        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator1, True)
        toggle_grad(self.discriminator2, False)
        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()
        self.d1_optimizer.zero_grad()

        with torch.no_grad():
            x_fake = self.generator(z)

        # On real data
        x_real.requires_grad_()
        d_real = self.discriminator1(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # Reg on real
        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            dloss_tradition = reg


        # On fake data
        x_fake.requires_grad_()
        d_fake = self.discriminator1(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        # Reg on fake
        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            dloss_tradition += reg

        # dual loss
        d1_loss_dual, _, _ = self._dual_loss(self.discriminator1, self.discriminator2, x_real.clone(), x_fake.clone(),
                                             z, detach_g=True)
        d1_loss_dual = d1_loss_dual * self.beta
        var_d1_loss_dual = Variable(d1_loss_dual.data, requires_grad=True)

        dloss_tradition = dloss_real + dloss_fake

        dloss = dloss_tradition + var_d1_loss_dual
        dloss.backward()
        self.d1_optimizer.step()

        return dloss_tradition.item(), d1_loss_dual.item()

    def discriminator2_trainstep(self, x_real, z):

        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator1, False)
        toggle_grad(self.discriminator2, True)
        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()
        self.d2_optimizer.zero_grad()

        with torch.no_grad():
            x_fake = self.generator(z)

        # On real data
        x_real.requires_grad_()
        d_real = self.discriminator2(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # Reg on real
        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            dloss_tradition = reg


        # On fake data
        x_fake.requires_grad_()
        d_fake = self.discriminator2(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        # Reg on fake
        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            dloss_tradition += reg

        # dual loss
        d2_loss_dual, _, _ = self._dual_loss(self.discriminator2, self.discriminator1, x_real.clone(), x_fake.clone(),
                                             z, detach_g=True)
        d2_loss_dual = d2_loss_dual * self.beta
        var_d2_loss_dual = Variable(d2_loss_dual.data, requires_grad=True)

        dloss_tradition = dloss_real + dloss_fake

        dloss = dloss_tradition + var_d2_loss_dual
        dloss.backward()

        self.d2_optimizer.step()

        return dloss_tradition.item(), d2_loss_dual.item()
    
    def d2gan_trainstep(self, x_real, z):
        ######################################
        # train D1 and D2
        #####################################
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator1, True)
        toggle_grad(self.discriminator2, True)
        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()
        self.d1_optimizer.zero_grad()
        self.d2_optimizer.zero_grad()

        with torch.no_grad():
            x_fake = self.generator(z)
            
        # D1 sees real as real, minimize -logD1(x)
        x_real.requires_grad_()
        d1_real = self.discriminator1(x_real)
        errD1_real = 0.2 * self.criterion_log(d1_real)
        # errD1_real = self.compute_loss(d1_real, 1)
        errD1_real.backward()
        

        # D2 sees real as fake, minimize D2(x)
        d2_real = self.discriminator2(x_real)
        errD2_real = self.criterion_itself(d2_real, False)
        errD2_real.backward()

        # D1 sees fake as fake, minimize D1(G(z))
        x_fake.requires_grad_()
        d1_fake = self.discriminator1(x_fake.detach())
        errD1_fake = self.criterion_itself(d1_fake, False)
        # errD1_fake = self.compute_loss(d1_fake, 0)
        errD1_fake.backward()

        # D2 sees fake as fake, minimize D1(G(z))
        d2_fake = self.discriminator2(x_fake.detach())
        errD2_fake = 0.1 * self.criterion_log(d2_fake)
        errD2_fake.backward()

        self.d1_optimizer.step()
        self.d2_optimizer.step()
        ##################################
        # train G
        ##################################
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator1, False)
        toggle_grad(self.discriminator2, False)

        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z)
        x_real.requires_grad_()
        x_fake.requires_grad_()

        d1_fake = self.discriminator1(x_fake)
        errG1 = self.criterion_itself(d1_fake)
        # errG1 = self.compute_loss(d1_fake, 1)

        d2_fake = self.discriminator2(x_fake)
        errG2 = self.criterion_log(d2_fake, False)

        errG = errG2*0.1 + errG1
        errG.backward()
        self.g_optimizer.step()
        
        return errG1.item(), errD1_real.item(), errD1_fake.item()
        # return errG.item(),errD1_real.item()+errD1_fake.item(),errD2_real.item()+errD2_fake.item()

    def dcgan_trainstep(self, x_real, z):
        ######################################
        # train D1 and D2
        #####################################
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator1, True)
        toggle_grad(self.discriminator2, False)
        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()
        self.d1_optimizer.zero_grad()

        with torch.no_grad():
            x_fake = self.generator(z)
            
        # D1 sees real as real, minimize -logD1(x)
        x_real.requires_grad_()
        d1_real = self.discriminator1(x_real)
        errD1_real = self.compute_loss(d1_real, 1)
        errD1_real.backward()
        

        # D1 sees fake as fake, minimize D1(G(z))
        x_fake.requires_grad_()
        d1_fake = self.discriminator1(x_fake.detach())
        errD1_fake = self.compute_loss(d1_fake, 0)
        errD1_fake.backward()

        self.d1_optimizer.step()
        ##################################
        # train G
        ##################################
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator1, False)
        toggle_grad(self.discriminator2, False)

        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z)
        x_real.requires_grad_()
        x_fake.requires_grad_()

        d1_fake = self.discriminator1(x_fake)
        errG = self.compute_loss(d1_fake, 1)

        errG.backward()
        self.g_optimizer.step()
        
        return errG.item(), errD1_real.item(), errD1_fake.item()

    def wgan_trainstep(self, x_real, z):
        ######################################
        # train D1 and D2
        #####################################
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator1, True)
        self.generator.train()
        self.discriminator1.train()
        self.d1_optimizer.zero_grad()

        with torch.no_grad():
            x_fake = self.generator(z)
            
        x_real.requires_grad_()
        x_fake.requires_grad_()
        gradient_penalty = self.compute_gradient_penalty( x_real.detach(), x_fake.detach())

        loss_D = -torch.mean(self.discriminator1(x_real)) + torch.mean(self.discriminator1(x_fake.detach())) \
                + 10 * gradient_penalty
        loss_D.backward()
        
        self.d1_optimizer.step()

        # Clip weights of discriminator
        for p in self.discriminator1.parameters():
            p.data.clamp_(-0.01, 0.01)
        ##################################
        # train G
        ##################################
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator1, False)

        self.generator.train()
        self.discriminator1.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z)
        x_fake.requires_grad_()

        loss_G = -torch.mean(self.discriminator1(x_fake))
        loss_G.backward()
        self.g_optimizer.step()
        
        return loss_G.item(), loss_D.item(), 0 

    def compute_gradient_penalty(self, real_samples, fake_samples):
        D = self.discriminator1
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0]).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def dcgan_w2d_trainstep(self, x_real, z):
        ######################################
        # train D1 and D2
        #####################################
        d1loss, _ = self.discriminator1_trainstep(x_real, z)
        d2loss, _ = self.discriminator2_trainstep(x_real, z)
        ##################################
        # train G
        ##################################
        gloss = self.generator_trainstep(x_real, z)
        
        return gloss, d1loss, d2loss

    def compute_loss(self, d_out, target, value=True):
        if value:
            targets = d_out.new_full(size=d_out.size(), fill_value=target)
        else:
            targets = target

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy(torch.sigmoid(d_out), targets)
        elif self.gan_type == 'wgan':
            loss = (2 * target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def adversarial_loss(self, d_out, target):
        loss = self.adversarial_loss_stable(torch.sigmoid(d_out), torch.sigmoid(target))
        return loss

    def choose_random_samples(self, shape, x_real, x_fake, z, detach_g):

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

    def _dual_loss(self, d_to_backprop, d_dual, x_real, x_fake, z, detach_g=False):
        dual_losses = []
        d = 1

        for _ in range(d):
            x1, x2, x3 = self.choose_random_samples((3, x_real.shape[0]), x_real, x_fake, z, detach_g)

            p1 = self.adversarial_loss(d_to_backprop(x1), d_dual(x1).detach())
            p2 = self.adversarial_loss(d_to_backprop(x2), d_dual(x3).detach())
            l = p1 - self.alpha * p2
            dual_losses.append(l)

        dual_loss = torch.mean(torch.stack(dual_losses))
        return dual_loss, p1, p2


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


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

class Log_loss(torch.nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(torch.nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss
