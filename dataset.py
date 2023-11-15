import itertools
import math
import pathlib

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from tqdm import tqdm


def repeater(iterable):
    for it in itertools.repeat(iterable):
        for item in it:
            yield item


class L3FDataset(Dataset):

    def __init__(self, cfg, mode, memorize=False):
        super().__init__()

        assert cfg.split in ['20', '50', '100']
        assert mode in ['train', 'test']

        root = pathlib.Path(cfg.root)
        assert root.exists()

        self.gt_fpaths = sorted(root.joinpath('jpeg', mode, '1').glob('*.jpg'))
        self.lq_fpaths = sorted(root.joinpath('jpeg', mode, f'1_{cfg.split}').glob('*.jpg'))

        self.mode = mode
        self.patch_size = cfg.patch_size or None

        self.training_view_size = cfg.view_size or None

        # crop central views images
        self.resolution = cfg.cropped_resolution or 15

        # ignore more views from the top and left boundary to align the official implementation
        self.sampled_area_start = math.ceil((15 - self.resolution) * 0.5)
        self.sampled_area_end = self.sampled_area_start + self.resolution

        self.memorize = memorize
        # memory cache, indexed by file paths
        self.all_views_dict = {}
        self.neighbor_views_dict = {}
        self.gt_views_dict = {}

        if memorize:
            for i in tqdm(range(len(self)), desc='loading dataset into memory', leave=False):
                _ = self[i]
            print('dataset has been loaded')

    def __len__(self):
        return len(self.gt_fpaths)

    def __getitem__(self, index):
        lq_fpath, gt_fpath = self.lq_fpaths[index], self.gt_fpaths[index]
        stem = lq_fpath.stem
        if self.memorize:
            if lq_fpath not in self.all_views_dict:
                all_views, neighbor_views, gt_views = self._pack_views(lq_fpath, gt_fpath)
                self.all_views_dict[lq_fpath] = all_views
                self.neighbor_views_dict[lq_fpath] = neighbor_views
                self.gt_views_dict[gt_fpath] = gt_views

            all_views = self.all_views_dict[lq_fpath]
            neighbor_views = self.neighbor_views_dict[lq_fpath]
            gt_views = self.gt_views_dict[gt_fpath]
        else:
            all_views, neighbor_views, gt_views = self._pack_views(lq_fpath, gt_fpath)

        # all views: [n_span_views, 3, img_H, img_W]
        # neighbor views: [n_span_views, 5, 3, img_H, img_W]
        # gt view: [n_span_views, 3, img_H, img_W]

        if self.mode == 'train':
            # Random Truncating
            if self.training_view_size:
                idxs = torch.randperm(neighbor_views.shape[0])[:self.training_view_size]
                out_neighbor_views, out_gt_views = neighbor_views[idxs], gt_views[idxs]
                out_all_views = all_views
            else:
                out_neighbor_views, out_gt_views, out_all_views = neighbor_views, gt_views, all_views

            # Random Cropping
            i, j, th, tw = RandomCrop.get_params(all_views, (self.patch_size, self.patch_size))
            out_all_views = T.crop(out_all_views, i, j, th, tw)
            out_neighbor_views = T.crop(out_neighbor_views, i, j, th, tw)
            out_gt_views = T.crop(out_gt_views, i, j, th, tw)

            # Random Flipping
            if torch.rand(1) < 0.5:
                out_all_views = T.hflip(out_all_views)
                out_neighbor_views = T.hflip(out_neighbor_views)
                out_gt_views = T.hflip(out_gt_views)

            if torch.rand(1) < 0.2:
                out_all_views = T.vflip(out_all_views)
                out_neighbor_views = T.vflip(out_neighbor_views)
                out_gt_views = T.vflip(out_gt_views)

            # Channel Permutation
            if torch.rand(1) < 0.5:
                perm_order = torch.randperm(3)
                out_all_views = out_all_views[:, perm_order]
                out_neighbor_views = out_neighbor_views[:, :, perm_order]
                out_gt_views = out_gt_views[:, perm_order]
        else:
            out_all_views = self._pad_test_image(all_views, 2)
            out_neighbor_views = self._pad_test_image(neighbor_views, 2)
            out_gt_views = gt_views

        return {'all': out_all_views, 'neighbor': out_neighbor_views, 'gt': out_gt_views, 'stem': stem}

    def _pack_views(self, lq_fpath, gt_fpath):
        lq_img = T.to_tensor(Image.open(lq_fpath))
        gt_img = T.to_tensor(Image.open(gt_fpath))

        lq_patches = rearrange(lq_img, 'c (n1 h) (n2 w) -> n1 n2 c h w', n1=15, n2=15)
        gt_patches = rearrange(gt_img, 'c (n1 h) (n2 w) -> n1 n2 c h w', n1=15, n2=15)

        sai_H, sai_W = lq_patches.shape[-2:]

        span, n_span_views = self.resolution, self.resolution ** 2
        packed_all_views, packed_neighbor_views, packed_gt_views = (
            torch.empty(size=(n_span_views, 3, sai_H, sai_W)),
            torch.empty(size=(n_span_views, 5, 3, sai_H, sai_W)),
            torch.empty(size=(n_span_views, 3, sai_H, sai_W))
        )
        for i, j in itertools.product(range(span), range(span)):
            row, col = self.sampled_area_start + i, self.sampled_area_start + j
            center_view = lq_patches[row, col]
            neighbor_views = [lq_patches[row+dr, col+dc] for (dr, dc) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]]
            gt_view = gt_patches[row, col]

            packed_all_views[i*span+j] = center_view
            packed_neighbor_views[i*span+j] = torch.stack(neighbor_views, dim=0)
            packed_gt_views[i*span+j] = gt_view

        return packed_all_views, packed_neighbor_views, packed_gt_views

    def _pad_test_image(self, img, size_divisibility):
        h, w = img.shape[-2:]
        new_h = ((h + size_divisibility) // size_divisibility) * size_divisibility
        new_w = ((w + size_divisibility) // size_divisibility) * size_divisibility
        pad_h = new_h - h if h % size_divisibility != 0 else 0
        pad_w = new_w - w if w % size_divisibility != 0 else 0
        out = F.pad(img, [0, pad_w, 0, pad_h, *(0, 0) * (img.ndim-4)], mode='reflect')
        return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from addict import Dict
    from mpl_toolkits.axes_grid1 import ImageGrid
    from torchvision.utils import make_grid

    cfg = Dict({'root': './L3F-dataset', 'split': '20', 'cropped_resolution': 8})
    dataset = L3FDataset(cfg, mode='test', memorize=False)

    fig = plt.figure(figsize=(20, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

    data = dataset[0]
    all, gt = (
        torch.clamp(data['all'] * 10, 0., 1.),
        data['gt']
    )

    all = make_grid(all, nrow=cfg.cropped_resolution, padding=0, normalize=False)
    gt = make_grid(gt, nrow=cfg.cropped_resolution, padding=0, normalize=False)

    all = all.squeeze(0).permute(1, 2, 0).numpy()
    gt = gt.squeeze(0).permute(1, 2, 0).numpy()

    for ax, img in zip(grid, [all, gt]):
        ax.imshow(img)

    plt.show()
    plt.close()
