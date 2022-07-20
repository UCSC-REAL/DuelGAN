import torch, argparse, sys, os, numpy
from .sampler import FixedRandomSubsetSampler, FixedSubsetSampler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms, utils
from . import pbar, zdataset, segmenter, frechet_distance, parallelfolder

NUM_OBJECTS = 336


def main():
    parser = argparse.ArgumentParser(description='Net dissect utility')
    parser.add_argument('true_dir')
    parser.add_argument('gen_dir')
    parser.add_argument('--size', type=int, default=10000)
    parser.add_argument('--cachedir', default='results/fsd/cache')
    parser.add_argument('--histout', default=None)
    parser.add_argument('--maxscale', type=float, default=50)
    parser.add_argument('--labelcount', type=int, default=30)
    parser.add_argument('--dpi', type=float, default=100)
    parser.add_argument('--it', type=str, default="-1")
    parser.add_argument('--results_dir', default=None, help='path to results_dir')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args.true_dir, args.gen_dir)
    true_dir, gen_dir = args.true_dir, args.gen_dir
    seed1, seed2 = [1, 1 if true_dir != gen_dir else 2]
    true_tally, gen_tally = [
        cached_tally_directory(d,
                               size=args.size,
                               cachedir=args.cachedir,
                               seed=seed)
        for d, seed in [(true_dir, seed1), (gen_dir, seed2)]
    ]
    fsd, meandiff, covdiff = frechet_distance.sample_frechet_distance(
        true_tally * 100, gen_tally * 100, return_components=True)
    print('fsd: %f; meandiff: %f; covdiff: %f' % (fsd, meandiff, covdiff))
    if args.histout is not None:
        diff_figure(true_tally * 100,
                    gen_tally * 100,
                    labelcount=args.labelcount,
                    maxscale=args.maxscale,
                    dpi=args.dpi).savefig(args.histout)

    if args.results_dir is not None:
        import json

        it = args.it
        results_dir = args.results_dir

        with open(os.path.join(args.results_dir, 'fsd_results.json')) as f:
            fsd_results = json.load(f)

        fsd_results[it] = (fsd, meandiff, covdiff)
        
        with open(os.path.join(args.results_dir, 'fsd_results.json'), 'w') as f:
            f.write(json.dumps(fsd_results))

        diff_figure(true_tally * 100,
                    gen_tally * 100,
                    labelcount=args.labelcount,
                    maxscale=args.maxscale,
                    dpi=args.dpi).savefig(os.path.join(args.results_dir, f'fsd_{it}.png'))
    
def cached_tally_directory(directory, size=10000, cachedir=None, seed=1):
    filename = '%s_segtally_%d.npy' % (directory, size)
    if seed != 1:
        filename = '%d_%s' % (seed, filename)
    if cachedir is not None:
        filename = os.path.join(cachedir, filename.replace('/', '_'))
    #load only if gt stats, or image directory
    if os.path.isfile(filename) and (not directory.endswith('.npz') or 'gt' in directory):
        return numpy.load(filename)
    os.makedirs(cachedir, exist_ok=True)
    result = tally_directory(directory, size, seed=seed)
    numpy.save(filename, result)
    return result


def tally_directory(directory, size=10000, seed=1):
    if directory.endswith('.npz'):
        with np.load(directory) as f:
            images = torch.from_numpy(f['fake'])
            images = images.permute(0, 3, 1, 2) #BHWC -> BCHW
            images = (images/127.5) - 1 #normalize in [-1, 1]
            images = torch.nn.functional.interpolate(images, size=(256, 256))
            print(images.shape, images.max(), images.min())
        dataset = TensorDataset(images)
    else:  
        dataset = parallelfolder.ParallelImageFolders(
            [directory],
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
    loader = DataLoader(dataset,
                        sampler=FixedRandomSubsetSampler(dataset,
                                                         end=size,
                                                         seed=seed),
                        batch_size=10,
                        pin_memory=True)
    upp = segmenter.UnifiedParsingSegmenter()
    labelnames, catnames = upp.get_label_and_category_names()
    result = numpy.zeros((size, NUM_OBJECTS), dtype=numpy.float)
    batch_result = torch.zeros(loader.batch_size,
                               NUM_OBJECTS,
                               dtype=torch.float).cuda()
    with torch.no_grad():
        batch_index = 0
        for [batch] in pbar(loader):
            seg_result = upp.segment_batch(batch.cuda())
            for i in range(len(batch)):
                batch_result[i] = (seg_result[i, 0].view(-1).bincount(
                    minlength=NUM_OBJECTS).float() /
                                   (seg_result.shape[2] * seg_result.shape[3]))
            result[batch_index:batch_index +
                   len(batch)] = (batch_result.cpu().numpy())
            batch_index += len(batch)
    return result


def tally_dataset_objects(dataset, size=10000):
    loader = DataLoader(dataset,
                        sampler=FixedRandomSubsetSampler(dataset, end=size),
                        batch_size=10,
                        pin_memory=True)
    upp = segmenter.UnifiedParsingSegmenter()
    labelnames, catnames = upp.get_label_and_category_names()
    result = numpy.zeros((size, NUM_OBJECTS), dtype=numpy.float)
    batch_result = torch.zeros(loader.batch_size,
                               NUM_OBJECTS,
                               dtype=torch.float).cuda()
    with torch.no_grad():
        batch_index = 0
        for [batch] in pbar(loader):
            seg_result = upp.segment_batch(batch.cuda())
            for i in range(len(batch)):
                batch_result[i] = (seg_result[i, 0].view(-1).bincount(
                    minlength=NUM_OBJECTS).float() /
                                   (seg_result.shape[2] * seg_result.shape[3]))
            result[batch_index:batch_index +
                   len(batch)] = (batch_result.cpu().numpy())
            batch_index += len(batch)
    return result


def tally_generated_objects(model, size=10000):
    zds = zdataset.z_dataset_for_model(model, size)
    loader = DataLoader(zds, batch_size=10, pin_memory=True)
    upp = segmenter.UnifiedParsingSegmenter()
    labelnames, catnames = upp.get_label_and_category_names()
    result = numpy.zeros((size, NUM_OBJECTS), dtype=numpy.float)
    batch_result = torch.zeros(loader.batch_size,
                               NUM_OBJECTS,
                               dtype=torch.float).cuda()
    with torch.no_grad():
        batch_index = 0
        for [zbatch] in pbar(loader):
            img = model(zbatch.cuda())
            seg_result = upp.segment_batch(img)
            for i in range(len(zbatch)):
                batch_result[i] = (seg_result[i, 0].view(-1).bincount(
                    minlength=NUM_OBJECTS).float() /
                                   (seg_result.shape[2] * seg_result.shape[3]))
            result[batch_index:batch_index +
                   len(zbatch)] = (batch_result.cpu().numpy())
            batch_index += len(zbatch)
    return result


def diff_figure(ttally,
                gtally,
                labelcount=30,
                labelleft=True,
                dpi=100,
                maxscale=50.0,
                legend=False):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    tresult, gresult = [t.mean(0) for t in [ttally, gtally]]
    upp = segmenter.UnifiedParsingSegmenter()
    labelnames, catnames = upp.get_label_and_category_names()
    x = []
    labels = []
    gen_amount = []
    change_frac = []
    true_amount = []
    for label in numpy.argsort(-tresult):
        if label == 0 or labelnames[label][1] == 'material':
            continue
        if tresult[label] == 0:
            break
        x.append(len(x))
        labels.append(labelnames[label][0].split()[0])
        true_amount.append(tresult[label].item())
        gen_amount.append(gresult[label].item())
        change_frac.append(
            (float(gresult[label] - tresult[label]) / tresult[label]))
        if len(x) >= labelcount:
            break
    fig = Figure(dpi=dpi, figsize=(1.4 + 5.0 * labelcount / 30, 4.0))
    FigureCanvas(fig)
    a1, a0 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
    a0.bar(x, change_frac, label='relative delta')
    a0.set_xticks(x)
    a0.set_xticklabels(labels, rotation='vertical')
    if labelleft:
        a0.set_ylabel('relative delta\n(gen - train) / train')
    a0.set_xlim(-1.0, len(x))
    a0.set_ylim([-1, 1.1])
    a0.grid(axis='y', antialiased=False, alpha=0.25)
    if legend:
        a0.legend(loc=2)
    prev_high = None
    for ix, cf in enumerate(change_frac):
        if cf > 1.15:
            if prev_high == (ix - 1):
                offset = 0.1
            else:
                offset = 0.0
                prev_high = ix
            a0.text(ix,
                    1.15 + offset,
                    '%.1f' % cf,
                    horizontalalignment='center',
                    size=6)

    a1.bar(x, true_amount, label='training')
    a1.plot(x, gen_amount, linewidth=3, color='red', label='generated')
    a1.set_yscale('log')
    a1.set_xlim(-1.0, len(x))
    a1.set_ylim(maxscale / 5000, maxscale)
    from matplotlib.ticker import LogLocator
    # a1.yaxis.set_major_locator(LogLocator(subs=(1,)))
    # a1.yaxis.set_minor_locator(LogLocator(subs=(1,), numdecs=10))
    # a1.yaxis.set_minor_locator(LogLocator(subs=(1,2,3,4,5,6,7,8,9)))
    # a1.yaxis.set_minor_locator(yminor_locator)
    if labelleft:
        a1.set_ylabel('mean area\nlog scale')
    if legend:
        a1.legend()
    a1.set_yticks([1e-2, 1e-1, 1.0, 1e+1])
    a1.set_yticks([
        a * b for a in [1e-2, 1e-1, 1.0, 1e+1]
        for b in range(1, 10) if maxscale / 5000 <= a * b <= maxscale
    ], True)  # minor ticks.
    a1.set_xticks([])
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main()
