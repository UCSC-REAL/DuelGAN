import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np
import random


def get_images(root, N):
    if False and os.path.exists(root + '.txt'):
        with open(os.path.exists(root + '.txt')) as f:
            files = f.readlines()
            random.shuffle(files)
            return files
    else:
        all_files = []
        for i, (dp, dn, fn) in enumerate(os.walk(os.path.expanduser(root))):
            for j, f in enumerate(fn):
                if j >= 1000:
                    break  # don't get whole dataset, just get enough images per class
                if f.endswith(('.png', '.webp', 'jpg', '.JPEG')):
                    all_files.append(os.path.join(dp, f))
        random.shuffle(all_files)
        return all_files


def pt_to_np(imgs):
    '''normalizes pytorch image in [-1, 1] to [0, 255]'''
    return (imgs.permute(0, 2, 3, 1).mul_(0.5).add_(0.5).mul_(255)).clamp_(0, 255).numpy()


def get_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_gt_samples(dataset, nimgs=50000):

    if dataset == 'stl':
        data = datasets.STL10(paths[dataset], transform=get_transform(sizes[dataset]))
        images = []
        for x, y in tqdm(data):
            images.append(x)
        return pt_to_np(torch.stack(images))

    if dataset == 'cifar100':
        data = datasets.CIFAR100(paths[dataset], transform=get_transform(sizes[dataset]))
        images = []
        for x, y in tqdm(data):
            images.append(x)
        return pt_to_np(torch.stack(images))

    if dataset == 'celebA':
        data = datasets.CelebA(paths[dataset], transform=get_transform(sizes[dataset]))
        images = []
        for x, y in tqdm(data):
            images.append(x)
        return pt_to_np(torch.stack(images))

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(sizes[dataset]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        data = datasets.MNIST(paths[dataset], transform=transform)
        images = []
        for x, y in tqdm(data):
            images.append(x)
        return pt_to_np(torch.stack(images))
    if dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(sizes[dataset]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        data = datasets.FashionMNIST(paths[dataset], transform=transform)
        images = []
        for x, y in tqdm(data):
            images.append(x)
        return pt_to_np(torch.stack(images))

    elif dataset != 'cifar':
        transform = get_transform(sizes[dataset])
        all_images = get_images(paths[dataset], nimgs)
        images = []
        for file_path in tqdm(all_images[:nimgs]):
            images.append(transform(Image.open(file_path).convert('RGB')))
        print(paths[dataset])
        return pt_to_np(torch.stack(images))
    else:
        data = datasets.CIFAR10(paths[dataset], transform=get_transform(sizes[dataset]))
        images = []
        for x, y in tqdm(data):
            images.append(x)
        return pt_to_np(torch.stack(images))


paths = {
    'imagenet': 'data/ImageNet',
    'places': 'data/Places365',
    'cifar': 'data/CIFAR',
    'cifar100': 'data/CIFAR100',
    'mnist': 'data/mnist',
    'celebA': 'data/celebA',
    'fashion_mnist': 'data/FashionMNIST',
    'VGG': 'data/VGG/train/train',
    'stl': 'data/stl10'
}

sizes = {'imagenet': 128, 'places': 128, 'cifar': 32, 'cifar100': 32, 'mnist': 32, 'celebA': 32, 'fashion_mnist': 32,
         'VGG': 64, 'stl': 32}

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Save a batch of ground truth train set images for evaluation')
    parser.add_argument('--cifar', action='store_true')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--places', action='store_true')
    parser.add_argument('--cifar100', action='store_true')
    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--celebA', action='store_true')
    parser.add_argument('--fashion_mnist', action='store_true')
    parser.add_argument('--VGG', action='store_true')
    parser.add_argument('--stl', action='store_true')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    if args.cifar:
        cifar_samples = get_gt_samples('cifar', nimgs=50000)
        np.savez('output/cifar_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
    if args.stl:
        stl_samples = get_gt_samples('stl', nimgs=50000)
        np.savez('output/stl_gt_imgs.npz', fake=stl_samples, real=stl_samples)
    if args.imagenet:
        imagenet_samples = get_gt_samples('imagenet', nimgs=50000)
        np.savez('output/imagenet_gt_imgs.npz', fake=imagenet_samples, real=imagenet_samples)
    if args.places:
        places_samples = get_gt_samples('places', nimgs=50000)
        np.savez('output/places_gt_imgs.npz', fake=places_samples, real=places_samples)
    if args.cifar100:
        cifar_samples = get_gt_samples('cifar100', nimgs=50000)
        np.savez('output/cifar_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
    if args.mnist:
        cifar_samples = get_gt_samples('mnist', nimgs=50000)
        np.savez('output/mnist_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
    if args.celebA:
        cifar_samples = get_gt_samples('celebA', nimgs=50000)
        np.savez('output/celebA_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
    if args.fashion_mnist:
        cifar_samples = get_gt_samples('fashion_mnist', nimgs=50000)
        np.savez('output/fashion_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
    if args.VGG:
        cifar_samples = get_gt_samples('VGG', nimgs=50000)
        np.savez('output/VGG_64_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
