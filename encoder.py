import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from dcnv3 import DCNv3_pytorch
from transform import (
     PatchEmbed, 
     to_channels_first,
     to_channels_last,
)
from decoder import UPerNet
import einops


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class DownsampleLayer(nn.Module):
    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * channels, 
            kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer, 
            'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels // 2,
            kernel_size=2, stride=2, bias=False)
        self.norm = build_norm_layer(channels // 2, norm_layer, 
            'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
            act_layer='GELU', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer='GELU', norm_layer='LN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(out_channels, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.norm2 = build_norm_layer(out_channels, norm_layer, 'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class InternImageLayer(nn.Module):
    def __init__(self, opt, core_op, channels, depth, groups, drop_path):
        super().__init__()
        self.groups = groups

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = opt.post_norm
        self.dcn = core_op(channels=channels, kernel_size=3, stride=1, pad=1, dilation=1, 
            group=groups, offset_scale=opt.offset_scale, act_layer=opt.activation, norm_layer=opt.normalization)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels, hidden_features=int(channels * opt.mlp_ratio),
                            act_layer=opt.activation, drop=opt.dropout)
        self.layer_scale = opt.layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(opt.layer_scale * torch.ones(channels), requires_grad=True)
            self.gamma2 = nn.Parameter(opt.layer_scale * torch.ones(channels), requires_grad=True)

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        return x


class InternImageBlock(nn.Module):
    def __init__(self, opt, core_op, channels, depth, groups, drop_path, downsample=True):
        super().__init__()
        self.depth = depth
        self.post_norm = opt.post_norm

        self.blocks = nn.ModuleList([
            InternImageLayer(opt, core_op, channels, depth, groups,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                    ) for i in range(depth)
        ])
        if not self.post_norm:
            self.norm = build_norm_layer(channels, 'LN')

        if downsample:
            self.downsample = DownsampleLayer(
                    channels=channels, norm_layer=opt.normalization) 
        else:
            self.upsample = UpsampleLayer(
                    channels=channels, norm_layer=opt.normalization)

    def forward(self, x, return_wo_downsample=False):
        for blk in self.blocks:
            x = blk(x)
        if not self.post_norm:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)
        else:
            x = self.upsample(x)

        if return_wo_downsample:
            return x, x_
        return x


class Encoder(nn.Module):
    def __init__(self, opt, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], **kwargs):
        super().__init__()
        self.core_op = DCNv3_pytorch
        self.num_classes = opt.num_classes
        self.num_levels = len(depths)
        self.depths = depths
        self.img_size = opt.img_size
        self.patch_size = opt.patch_size
        self.in_channels = opt.in_channels
        self.num_features = int(opt.channels * 2 **(self.num_levels - 1))

        self.stem = StemLayer(in_channels=opt.in_channels, out_channels=opt.channels,
            act_layer=opt.activation, norm_layer=opt.normalization)
        self.pos_drop = nn.Dropout(p=opt.dropout)

        dpr = [
            x.item() for x in torch.linspace(0, opt.drop_path_rate, sum(depths))
        ]
        if opt.drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = opt.drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = InternImageBlock(opt, core_op=self.core_op, channels=int(opt.channels * 2**i),
                depth=depths[i], groups=groups[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=True)
            self.levels.append(level)

        self.conv_head = nn.Sequential(
            nn.Conv2d(self.num_features, int(self.num_features * opt.cls_scale),
                kernel_size=1, bias=False),
            build_norm_layer(int(self.num_features * opt.cls_scale), 'BN', 'channels_first', 'channels_first'),
            build_act_layer(opt.activation))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Conv2d(opt.num_classes, opt.num_classes, 3, 1, 1)
        self.upper = UPerNet(
            num_class=opt.num_classes, fc_dim=opt.channels * 8, use_softmax=False, fpn_dim=256)

        self.patch_embed = PatchEmbed(
                img_size=opt.img_size, patch_size=opt.patch_size, 
                frames=opt.in_channels, embed_dim=opt.embed_dim)
        self.input_size = self.patch_embed.input_size
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.input_size[1] * self.input_size[2], opt.embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.input_size[0], opt.embed_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, opt.embed_dim))
        self.embed = nn.Linear(opt.embed_dim, opt.embed_dim, bias=True)
        self.norm = build_norm_layer(opt.embed_dim, 'LN')

        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        

    def _init_deform_weights(self, m):
        if isinstance(m, self.core_op):
            m._reset_parameters()

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = 'encoder.{}.blocks.{}.'.format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios['levels.0.blocks.0.']
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios['levels.1.blocks.0.']
        lr_ratios["levels.0.norm"] = lr_ratios['levels.1.blocks.0.']
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios['levels.2.blocks.0.']
        lr_ratios["levels.1.norm"] = lr_ratios['levels.2.blocks.0.']
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios['levels.3.blocks.0.']
        lr_ratios["levels.2.norm"] = lr_ratios['levels.3.blocks.0.']
        return lr_ratios


    def random_masking(self, x, mask_ratio):
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def feature_embed(self, x, mask_ratio=0.75):
        # Patchify
        H, W  = self.img_size
        u, p1, p2 = self.patch_size
        t, h, w =  self.in_channels // u, H // p1, W // p2
        B, T, L, C = x.shape
        x = x.reshape(B, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio=mask_ratio)
        x = self.embed(x)
        C = x.shape[-1]
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, t * h * w + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x_ = x_.view([B, t * h * w, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )   
        x = x_.view([B, t * h * w, C])

        pos_embed = self.pos_embed_spatial.repeat(
            1, self.input_size[0], 1
        ) + torch.repeat_interleave(
            self.pos_embed_temporal,
            self.input_size[1] * self.input_size[2],
            dim=1,
        ) 
        x = x + pos_embed
        x = self.norm(x)

        # Un patchify
        x = x.reshape(shape=(B, t, h, w, u, p1, p2))
        x = torch.einsum("bthwupq->btuhpwq", x)
        imgs = x.reshape(shape=(B, -1, H, W))
        return imgs, mask


    def forward_features_seq_out(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x, mask =  self.feature_embed(x, mask_ratio)
        x = self.stem(x)
        x = self.pos_drop(x)

        seq_out = []
        for level in self.levels:
            x, x_ = level(x, return_wo_downsample=True)
            x_ = einops.rearrange(x_, "b h w c -> b c h w")
            seq_out.append(x_)
        return seq_out, mask


    def patchify(self, x):
        B, T, H, W = x.shape
        u, p1, p2 = self.patch_size
        assert H % p1 == 0 and W % p2 == 0 and T % u == 0
        h = H // p1
        w = W // p2
        t = T // u

        x = x.reshape(shape=(B, 1, t, u, h, p1, w, p2))
        x = torch.einsum("bctuhpwq->bthwupqc", x)
        x = x.reshape(shape=(B, t * h * w, u * p1 * p2))
        return x


    def forward_loss(self, imgs, pred, mask):
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.in_channels,
            )
            .long()
            .to(imgs.device),
        )
        imgs = self.patchify(_imgs.reshape(pred.shape))
        pred = self.patchify(pred)
        loss = (pred - imgs) ** 2
        loss = loss.mean(dim=-1)
        mask = mask.view(loss.shape)
        loss = (loss * mask).sum() / mask.sum()
        return loss


    def forward(self, x):
        imgs = x
        x, mask = self.forward_features_seq_out(x)
        x = self.upper(x, segSize=x[0].shape[1:])
        x = F.interpolate(x, imgs.shape[3:], mode='bilinear', align_corners=False)
        x = self.out(x)
        loss = self.forward_loss(imgs, x, mask)
        return loss, x

