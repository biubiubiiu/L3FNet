import torch
import torch.nn.functional as F
import torchvision.models as tvm


def l1_norm(params):
    return sum(torch.abs(p).sum() for p in params)


def contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.

    Args:
        x: features of shape (N, C, H, W).
        y: features of shape (N, C, H, W).

    Returns:
        cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    N, C, _, _ = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)

    d = 1 - cosine_sim                                  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
    d_min, _ = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)

    # Eq (2)
    d_tilde = d / (d_min + 1e-5)

    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)

    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)

    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, feature_layers, weights, distance=F.l1_loss, resize=True, normalized=True):
        super(VGGPerceptualLoss, self).__init__()

        assert len(feature_layers) == len(weights)

        self.feature_layers_weights = dict(zip(feature_layers, weights))

        self.features = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features[:max(feature_layers)]
        self.features.requires_grad_(False)
        self.features.eval()

        self.distance = distance

        self.resize = resize
        self.normalized = normalized
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if self.normalized:
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std

        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0
        x, y = input, target
        for i, block in enumerate(self.features):
            x, y = block(x), block(y)
            if i in self.feature_layers_weights:
                loss += self.distance(x, y) * self.feature_layers_weights[i]
        return loss
