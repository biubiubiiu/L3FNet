
import argparse
import logging
import os
from collections import defaultdict
from pprint import pformat

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

import losses
import utils
from dataset import L3FDataset, repeater
from net import L3FNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration')
    parser.add_argument('--resume_from', type=str, help='Resume training from existing checkpoint')
    parser.add_argument('--save_images', action='store_true', help='Dump predicted images')
    parser.add_argument('--no_save_images', dest='save_images', action='store_false')
    parser.set_defaults(save_images=True)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = utils.parse_config(args.config)
    env = utils.init_env(args, config)
    logging.info(f'using config file:\n{pformat(config)}')
    logging.info(f'using device {env.device}')

    model = L3FNet(resolution=config.net.resolution).to(env.device)
    optimizer = Adam(model.parameters(), lr=config.train.base_lr)

    train_dataset = L3FDataset(config.train.dataset, mode='train', memorize=True)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                              shuffle=True, num_workers=config.env.num_workers)
    train_loader = repeater(train_loader)  # infinite sampling

    val_dataset = L3FDataset(config.val.dataset, mode='test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    val_metrics = [utils.init_metrics(it) for it in config.val.metrics]
    primary_val_metric_idx = next((i for i, m in enumerate(config.val.metrics) if m.primary == True), 0)
    primary_metric_best = -999  # assume that the higher, the better

    writer = SummaryWriter(log_dir=os.path.join(env.save_dir, 'tensorboard'))

    start_iter = 1
    if args.resume_from:
        logging.info(f'resume training from {args.resume_from}')
        ckpt = torch.load(args.resume_from, map_location=env.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = int(ckpt['start_iter'])

    acc_loss_dict = defaultdict(lambda: 0)
    vgg_context_loss = losses.VGGPerceptualLoss(
        feature_layers=[9, 13, 18],
        weights=[0.1, 0.15, 0.15],
        distance=losses.contextual_loss,
        resize=False, normalized=False).to(env.device)
    for iteration in tqdm(range(start_iter, config.train.num_iters+1), dynamic_ncols=True, desc='Training'):
        model.train()
        data = next(train_loader)
        all, neighbor, gt = (
            data['all'].squeeze(0).to(env.device),
            data['neighbor'].squeeze(0).to(env.device),
            data['gt'].squeeze(0).to(env.device)
        )
        out = model(all, neighbor)

        l1_loss = F.l1_loss(out, gt) * (5 if iteration < 20000 else 1)
        context_loss = vgg_context_loss(out, gt) * 0.1
        l1_norm = losses.l1_norm(model.parameters()) * 1e-6
        loss = l1_loss + context_loss + l1_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss_dict['l1_loss'] += l1_loss.item()
        acc_loss_dict['context_loss'] += context_loss.item()
        acc_loss_dict['l1_norm'] += l1_norm.item()

        if iteration % len(train_dataset) == 0:
            logging.debug(
                f'[Iter {iteration}] ' + ', '.join([f'{name}: {val:.3f}' for name, val in acc_loss_dict.items()]))
            for name, val in acc_loss_dict.items():
                writer.add_scalar(name, val, iteration)
            acc_loss_dict.clear()

        if iteration % config.val.val_step == 0:
            model.eval()
            with torch.inference_mode():
                for data in tqdm(val_loader, leave=False, dynamic_ncols=True, desc=f'Evaluating'):
                    all, neighbor, gt, stem = (
                        data['all'].squeeze(0).to(env.device),
                        data['neighbor'].squeeze(0).to(env.device),
                        data['gt'].squeeze(0).to(env.device),
                        data['stem'][0]
                    )
                    out = model(all, neighbor)

                    # crop back to original shape
                    h, w = gt.shape[-2:]
                    out = out[..., :h, :w]

                    quant_out = out.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
                    quant_gt = gt.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)

                    for (_, metric) in val_metrics:
                        metric.update(quant_out.float(), quant_gt.float())

                    if args.save_images:
                        save_path = os.path.join(env.visual_dir(iter='final'), f'{stem}.png')
                        save_image(out, save_path, nrow=config.net.resolution, padding=0, normalize=False)

                metric_vals = []
                for i, (name, metric) in enumerate(val_metrics):
                    metric_val = metric.compute()
                    metric_vals.append(metric_val)

                    writer.add_scalar(f'{name.lower()}', metric_val)

                    if i == primary_val_metric_idx and metric_val > primary_metric_best:
                        primary_metric_best = metric_val
                        ckpt_path = os.path.join(env.save_dir, 'best.pth')
                        utils.save_state_dict(model, optimizer, iteration, ckpt_path)
                        logging.debug(f'save checkpoint to {ckpt_path} with {name}={metric_val}')

                    # reset internal state such that metric ready for new data
                    metric.reset()

                summary = '; '.join([f'{name} {val:.3f}' for (name, _), val in zip(val_metrics, metric_vals)])
                logging.debug(f'iter {iteration} validation: {summary}')

        if config.train.save_step and iteration % config.train.save_step == 0:
            utils.save_state_dict(model, optimizer, iteration, os.path.join(env.save_dir, f'iter{iteration}.pth'))

        utils.save_state_dict(model, optimizer, iteration, os.path.join(env.save_dir, 'latest.pth'))


if __name__ == '__main__':
    main()
