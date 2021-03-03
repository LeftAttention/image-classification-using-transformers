import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .utils import PreNorm, FeedForward, Attention

class TNT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_dim,
        pixel_dim,
        patch_size,
        pixel_size,
        depth,
        num_classes,
        heads = 8,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'image size must be divisible by patch size'

        num_patch_tokens = (image_size // patch_size) ** 2
        pixel_width = patch_size // pixel_size
        num_pixels = pixel_width ** 2

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))

        self.to_pixel_tokens = nn.Sequential(
            Rearrange('b c (p1 h) (p2 w) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size),
            nn.Unfold(pixel_width, stride = pixel_width),
            Rearrange('... c n -> ... n c'),
            nn.Linear(3 * pixel_width ** 2, pixel_dim)
        )

        self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
        self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))

        layers = nn.ModuleList([])
        for _ in range(depth):

            pixel_to_patch = nn.Sequential(
                nn.LayerNorm(pixel_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(pixel_dim * num_pixels, patch_dim),
            )

            layers.append(nn.ModuleList([
                PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = ff_dropout)),
                pixel_to_patch,
                PreNorm(patch_dim, Attention(dim = patch_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim = patch_dim, dropout = ff_dropout)),
            ]))

        self.layers = layers

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, num_classes)
        )

    def forward(self, x):
        b, _, h, w, patch_size, image_size = *x.shape, self.patch_size, self.image_size
        assert h == image_size and w == image_size, f'height {h} and width {w} of input must be given image size of {image_size}'

        num_patches = image_size // patch_size

        pixels = self.to_pixel_tokens(x)
        patches = repeat(self.patch_tokens, 'n d -> b n d', b = b)

        patches += rearrange(self.patch_pos_emb, 'n d -> () n d')
        pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

        for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:

            pixels = pixel_attn(pixels) + pixels
            pixels = pixel_ff(pixels) + pixels

            patches_residual = pixel_to_patch_residual(pixels)

            patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches, w = num_patches)
            patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
            patches = patches + patches_residual

            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches

        cls_token = patches[:, 0]
        return self.mlp_head(cls_token)
