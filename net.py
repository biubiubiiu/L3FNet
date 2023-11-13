import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        return out + x


class L3FNet(nn.Module):
    def __init__(self, resolution=8):
        super(L3FNet, self).__init__()

        # stage 1
        self.grb = nn.Sequential(
            nn.Conv2d(in_channels=3*resolution*resolution, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            *[ResBlock(channels=128) for _ in range(4)],
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
        )

        # stage2
        self.neighbor_views_encoder = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(in_channels=15, out_channels=64, kernel_size=3, stride=2, padding=1),
        )
        self.vrb = nn.Sequential(
            *[ResBlock(channels=128) for _ in range(6)],
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, all_views, neighbor_views):
        all_views = rearrange(all_views, 'n c h w -> 1 (n c) h w')
        neighbor_views = rearrange(neighbor_views, 'n r c h w -> n (r c) h w')
        centered_view = neighbor_views[:, :3].clone()

        global_repr = self.grb(all_views).repeat(neighbor_views.shape[0], *(1,) * (neighbor_views.ndim - 1))
        neighbor_views_feat = self.neighbor_views_encoder(neighbor_views)
        out = self.vrb(torch.cat([global_repr, neighbor_views_feat], dim=1))
        out = out + centered_view
        return out


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    # calculate macs and params
    net = L3FNet()
    x1 = torch.randn(64, 3, 64, 64)
    x2 = torch.randn(12, 5, 3, 64, 64)
    flops = FlopCountAnalysis(net, (x1, x2))
    with open(f'summary.txt', 'w', encoding='utf-8') as f:
        f.write(flop_count_table(flops))

    out = net(x1, x2)
    print(out.shape)
