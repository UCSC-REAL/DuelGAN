import argparse
import os
import copy
import pprint
from os import path

import torch
import numpy as np
from torch import nn

from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, get_clusterer, build_models, build_optimizers)
from seeing.pidfile import exit_if_job_done, mark_job_done

import time
from torch.autograd import Variable


torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
parser.add_argument('--nepochs', type=int, default=250, help='number of epochs to run before terminating')
parser.add_argument('--model_it', type=int, default=-1, help='which model iteration to load from, -1 loads the most recent model')
parser.add_argument('--devices', nargs='+', type=str, default=['0'], help='devices to use')

args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir'] if args.outdir is None else args.outdir


def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint({
        'data': config['data'],
        'generator': config['generator'],
        'discriminator1': config['discriminator1'],
        'discriminator2': config['discriminator2'],
        'clusterer': config['clusterer'],
        'training': config['training']
    })
    is_cuda = torch.cuda.is_available()

    # Short hands
    batch_size = config['training']['batch_size']
    log_every = config['training']['log_every']
    inception_every = config['training']['inception_every']
    backup_every = config['training']['backup_every']
    sample_nlabels = config['training']['sample_nlabels']
    nlabels = config['data']['nlabels']
    sample_nlabels = min(nlabels, sample_nlabels)

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    device = torch.device("cuda:0" if is_cuda else "cpu")
    CUDA = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    
    train_dataset, _ = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'],
        size=config['data']['img_size'],
        deterministic=config['data']['deterministic'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    # Create models
    generator, discriminator1, discriminator2 = build_models(config)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator1 = discriminator1.to(device)
    discriminator2 = discriminator2.to(device)

    for name, module in discriminator1.named_modules():
        if isinstance(module, nn.Sigmoid):
            print('Found sigmoid layer in discriminator; not compatible with BCE with logits')
            exit()

    for name, module in discriminator2.named_modules():
        if isinstance(module, nn.Sigmoid):
            print('Found sigmoid layer in discriminator; not compatible with BCE with logits')
            exit()

    g_optimizer, d1_optimizer, d2_optimizer = build_optimizers(generator, discriminator1, discriminator2, config)

    devices = [int(x) for x in args.devices]
    generator = nn.DataParallel(generator, device_ids=devices)
    discriminator1 = nn.DataParallel(discriminator1, device_ids=devices)
    discriminator2 = nn.DataParallel(discriminator2, device_ids=devices)

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=generator,
                                   discriminator1=discriminator1,
                                   discriminator2=discriminator2,
                                   g_optimizer=g_optimizer,
                                   d1_optimizer=d1_optimizer,
                                   d2_optimizer=d2_optimizer)

    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    monitoring=config['training']['monitoring'],
                    monitoring_dir=path.join(out_dir, 'monitoring'))

    # Distributions
    ydist = get_ydist(nlabels, device=device)
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    ntest = config['training']['ntest']
    x_test, y_test = utils.get_nsamples(train_loader, ntest)
    
    x_cluster, y_cluster = utils.get_nsamples(train_loader, config['clusterer']['nimgs'])
    x_test, y_test = x_test.to(device), y_test.to(device)
    z_test = zdist.sample((ntest, ))
    utils.save_images(x_test, path.join(out_dir, 'real.png'))
    logger.add_imgs(x_test, 'gt', 0)


    generator_test = generator

    clusterer = get_clusterer(config)(discriminator=discriminator1,
                                      x_cluster=x_cluster,
                                      x_labels=y_cluster,
                                      gt_nlabels=config['data']['nlabels'],
                                      **config['clusterer']['kwargs'])

    # Load checkpoint if it exists
    it = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it == -1 else args.model_it
    it, epoch_idx, loaded_clusterer = checkpoint_io.load_models(it=it, load_samples='supervised' != config['clusterer']['name'])

    if loaded_clusterer is None:
        print('Initializing new clusterer. The first clustering can be quite slow.')
        clusterer.recluster(discriminator=discriminator1)
        checkpoint_io.save_clusterer(clusterer, it=0)
        np.savez(os.path.join(checkpoint_dir, 'cluster_samples.npz'), x=x_cluster)
    else:
        print('Using loaded clusterer')
        clusterer = loaded_clusterer

    # Evaluator
    evaluator = Evaluator(
        generator_test,
        zdist,
        ydist,
        train_loader=train_loader,
        clusterer=clusterer,
        batch_size=batch_size,
        device=device,
        inception_nsamples=config['training']['inception_nsamples'])

    # Trainer
    trainer = Trainer(generator,
                      discriminator1,
                      discriminator2,
                      g_optimizer,
                      d1_optimizer,
                      d2_optimizer,
                      gan_type=config['training']['gan_type'],
                      reg_type=config['training']['reg_type'],
                      reg_param=config['training']['reg_param'],
                      train_loader= train_loader,
                      Tensor=Tensor,
                      dim=config['z_dist']['dim'],
                      alpha=0.0,
                      beta=0.0)

    # Training loop
    print('Start training...')
    time_mark = time.time()
    while it < args.nepochs * len(train_loader):
        epoch_idx += 1

        for x_real, y in train_loader:
            it += 1

            x_real, y = x_real.to(device), y.to(device)
            z = zdist.sample((batch_size, ))
            y = clusterer.get_labels(x_real, y).to(device)

            # dualLoss param adjust
            dual_alpha, dual_beta = trainer.update_dual_param(epoch_idx,args.nepochs,'all',config['training'])

            # Discriminator updates
            d1loss, dual_D1 = trainer.discriminator1_trainstep(x_real, z)
            logger.add('losses', 'discriminator1', d1loss, it=it)
            logger.add('losses', 'dual_D1', dual_D1, it=it)

            d2loss, dual_D2 = trainer.discriminator2_trainstep(x_real, z)
            logger.add('losses', 'discriminator2', d2loss, it=it)
            logger.add('losses', 'dual_D2', dual_D2, it=it)

            # Generators updates
            gloss = trainer.generator_trainstep(x_real, z)
            logger.add('losses', 'generator', gloss, it=it)


            # D2GAN training
            # gloss, d1loss, d2loss = trainer.d2gan_trainstep(x_real, z)
            # logger.add('losses', 'generator', gloss, it=it)
            # logger.add('losses', 'discriminator1', d1loss, it=it)
            # logger.add('losses', 'discriminator2', d2loss, it=it)
            
            # # DCGAN training
            # gloss, d1loss, d2loss = trainer.dcgan_trainstep(x_real, z)
            # logger.add('losses', 'generator', gloss, it=it)
            # logger.add('losses', 'discriminator1', d1loss, it=it)
            # logger.add('losses', 'discriminator2', d2loss, it=it)

            # # WGAN training
            # gloss, d1loss, d2loss = trainer.wgan_trainstep(x_real, z)
            # logger.add('losses', 'generator', gloss, it=it)
            # logger.add('losses', 'discriminator1', d1loss, it=it)
            # logger.add('losses', 'discriminator2', d2loss, it=it)

                        
            # DCGAN with 2 d training
            # gloss, d1loss, d2loss = trainer.dcgan_w2d_trainstep(x_real, z)
            # logger.add('losses', 'generator', gloss, it=it)
            # logger.add('losses', 'discriminator1', d1loss, it=it)
            # logger.add('losses', 'discriminator2', d2loss, it=it)

            
            


            # Print stats
            if it % log_every == 0:
                time_eplase = time.time() - time_mark
                time_mark = time.time()
                
                g_loss_last = logger.get_last('losses', 'generator')
                d1_loss_last = logger.get_last('losses', 'discriminator1')
                # d1_dual_last = logger.get_last('losses', 'dual_D1')
                d2_loss_last = logger.get_last('losses', 'discriminator2')
                # d2_dual_last = logger.get_last('losses', 'dual_D2')
                # print('[epoch %0d, it %4d, time %4f] g_loss = %.4f, d1_loss = %.4f, D1_dual=%.4f, d2_loss = %.4f, D2_dual=%.4f'
                #       % (epoch_idx, it, time_eplase, g_loss_last, d1_loss_last, d1_dual_last, d2_loss_last, d2_dual_last))
                # print('dual_alpha %.4f, dual_beta %.4f' % (dual_alpha, dual_beta))

                print('[epoch %0d, it %4d, time %4f] g_loss = %.4f, d1_loss = %.4f, d2_loss = %.4f'
                      % (epoch_idx, it, time_eplase, g_loss_last, d1_loss_last, d2_loss_last))


            # (i) Sample if necessary
            if it % config['training']['sample_every'] == 0:
                print('Creating samples...')
                x = evaluator.create_samples(z_test, y_test)
                x = evaluator.create_samples(z_test, clusterer.get_labels(x_test, y_test).to(device))
                logger.add_imgs(x, 'all', it)

                for y_inst in range(sample_nlabels):
                    x = evaluator.create_samples(z_test, y_inst)
                    logger.add_imgs(x, '%04d' % y_inst, it)

            # (ii) Backup if necessary
            if it % backup_every == 0:
                checkpoint_io.save('model_%08d.pt' % it, it=it)
                checkpoint_io.save_clusterer(clusterer, int(it))
                logger.save_stats('stats_%08d.p' % it)

                if it > 0:
                    checkpoint_io.save('model.pt', it=it)


if __name__ == '__main__':
    exit_if_job_done(out_dir)
    main()
    mark_job_done(out_dir)
