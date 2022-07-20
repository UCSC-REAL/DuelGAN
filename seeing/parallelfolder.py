'''
Variants of pytorch's ImageFolder for loading image datasets with more
information, such as parallel feature channels in separate files,
cached files with lists of filenames, etc.
'''

import os, torch, re, random, numpy, itertools
import torch.utils.data as data
from torchvision.datasets.folder import default_loader as tv_default_loader
from PIL import Image
from collections import OrderedDict
from . import pbar

def grayscale_loader(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('L')

class ndarray(numpy.ndarray):
    '''
    Wrapper to make ndarrays into heap objects so that shared_state can
    be attached as an attribute.
    '''
    pass

def default_loader(filename):
    '''
    Handles both numpy files and image formats.
    '''
    if filename.endswith('.npy'):
        return numpy.load(filename).view(ndarray)
    elif filename.endswith('.npz'):
        return numpy.load(filename)
    else:
        return tv_default_loader(filename)

class ParallelImageFolders(data.Dataset):
    """
    A data loader that looks for parallel image filenames, for example

    photo1/park/004234.jpg
    photo1/park/004236.jpg
    photo1/park/004237.jpg

    photo2/park/004234.png
    photo2/park/004236.png
    photo2/park/004237.png
    """
    def __init__(self, image_roots,
            transform=None,
            loader=default_loader,
            stacker=None,
            classification=False,
            intersection=False,
            filter_tuples=None,
            verbose=None,
            size=None,
            shuffle=None,
            lazy_init=True):
        self.image_roots = image_roots
        if transform is not None and not hasattr(transform, '__iter__'):
            transform = [transform for _ in image_roots]
        self.transforms = transform
        self.stacker = stacker
        self.loader = loader
        def do_lazy_init():
            self.images, self.classes, self.class_to_idx = (
                    make_parallel_dataset(image_roots,
                        classification=classification,
                        intersection=intersection,
                        filter_tuples=filter_tuples,
                        verbose=verbose))
            if len(self.images) == 0:
                raise RuntimeError("Found 0 images within: %s" % image_roots)
            if shuffle is not None:
                random.Random(shuffle).shuffle(self.images)
            if size is not None:
                self.image = self.images[:size]
            self._do_lazy_init = None
        # Do slow initialization lazily.
        if lazy_init:
            self._do_lazy_init = do_lazy_init
        else:
            do_lazy_init()

    def __getattr__(self, attr):
        if self._do_lazy_init is not None:
            self._do_lazy_init()
            return getattr(self, attr)
        raise AttributeError()

    def __getitem__(self, index):
        if self._do_lazy_init is not None:
            self._do_lazy_init()
        paths = self.images[index]
        if self.classes is not None:
            classidx = paths[-1]
            paths = paths[:-1]
        sources = [self.loader(path) for path in paths]
        # Add a common shared state dict to allow random crops/flips to be
        # coordinated.
        shared_state = {}
        for s in sources:
            try:
                s.shared_state = shared_state
            except:
                pass
        if self.transforms is not None:
            sources = [transform(source) if transform is not None else source
                    for source, transform
                    in itertools.zip_longest(sources, self.transforms)]
        if self.stacker is not None:
            sources = self.stacker(sources)
            if self.classes is not None:
                sources = (sources, classidx)
        else:
            if self.classes is not None:
                sources.append(classidx)
            sources = tuple(sources)
        return sources

    def __len__(self):
        if self._do_lazy_init is not None:
            self._do_lazy_init()
        return len(self.images)

def is_npy_file(path):
    return path.endswith('.npy') or path.endswith('.NPY')

def is_image_file(path):
    return None != re.search(r'\.(jpe?g|png)$', path, re.IGNORECASE)

def walk_image_files(rootdir, verbose=None):
    indexfile = '%s.txt' % rootdir
    if os.path.isfile(indexfile):
        basedir = os.path.dirname(rootdir)
        with open(indexfile) as f:
            result = sorted([os.path.join(basedir, line.strip())
                for line in f.readlines()])
            return result
    result = []
    for dirname, _, fnames in sorted(pbar(os.walk(rootdir),
            desc='Walking %s' % os.path.basename(rootdir))):
        for fname in sorted(fnames):
            if is_image_file(fname) or is_npy_file(fname):
                result.append(os.path.join(dirname, fname))
    return result

def make_parallel_dataset(image_roots, classification=False,
        intersection=False, filter_tuples=None, verbose=None):
    """
    Returns ([(img1, img2, clsid), (img1, img2, clsid)..],
             classes, class_to_idx)
    """
    image_roots = [os.path.expanduser(d) for d in image_roots]
    image_sets = OrderedDict()
    for j, root in enumerate(image_roots):
        for path in walk_image_files(root, verbose=verbose):
            key = os.path.splitext(os.path.relpath(path, root))[0]
            if key not in image_sets:
                image_sets[key] = []
            if not intersection and len(image_sets[key]) != j:
                raise RuntimeError(
                    'Images not parallel: %s missing from one dir' % (key))
            image_sets[key].append(path)
    if classification:
        classes = sorted(set([os.path.basename(os.path.dirname(k))
            for k in image_sets.keys()]))
        class_to_idx = dict({k: v for v, k in enumerate(classes)})
        for k, v in image_sets.items():
            v.append(class_to_idx[os.path.basename(os.path.dirname(k))])
    else:
        classes, class_to_idx = None, None
    tuples = []
    for key, value in image_sets.items():
        if len(value) != len(image_roots) + (1 if classification else 0):
            if intersection:
                continue
            else:
                raise RuntimeError(
                    'Images not parallel: %s missing from one dir' % (key))
        value = tuple(value)
        if filter_tuples and not filter_tuples(value):
            continue
        tuples.append(value)
    return tuples, classes, class_to_idx

