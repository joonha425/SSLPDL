import logging
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

logger = logging.getLogger(__name__)

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size, patch_size, frames=16, embed_dim=1024, in_chans=1):
        super().__init__()
        assert img_size[1] % patch_size[2] == 0
        assert img_size[0] % patch_size[1] == 0
        assert frames % patch_size[0] == 0
        num_patches = (
            (img_size[1] // patch_size[2])
            * (img_size[0] // patch_size[1])
            * (frames // patch_size[0])
        )
        self.input_size = (
            frames // patch_size[0],
            img_size[0] // patch_size[1],
            img_size[1] // patch_size[2],
        )
        self.img_size = img_size
        self.frames = frames
        self.num_patches = num_patches
        self.grid_size = (img_size[0] // patch_size[1], img_size[1] // patch_size[2])

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.frames
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)




