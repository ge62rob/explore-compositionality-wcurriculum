# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

#added for hybrid loss
from pytorch_msssim import ssim
import torch.nn.functional as F


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone - Grayscale version (1 channel)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,  # Changed from 3 to 1
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 * 1)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))  # Changed from 3 to 1
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))  # Changed from p**2 * 3 to p**2 * 1
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))  # Changed from 3 to 1
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))  # Changed from 3 to 1
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    '''
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    '''
    '''
    def forward_loss_hybrid(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]  # raw decoder logits
        mask: [N, L], 0 = keep, 1 = remove
        """
        # 0) bound your predictions into [0,1]
        pred = torch.sigmoid(pred)

        # 1) MSE on the masked patches
        target = self.patchify(imgs)                       # [N, L, p*p]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        mse = (pred - target).pow(2).mean(dim=-1)           # [N, L]
        mse_loss = (mse * mask).sum() / mask.sum()

        # 2) SSIM on the full reconstruction
        recon      = self.unpatchify(pred).clamp(0., 1.)    # [N, 1, H, W]
        ssim_score = ssim(recon, imgs, data_range=1.0, size_average=True)
        ssim_loss  = 1.0 - ssim_score

        # 3) Hybrid combine loss
        alpha, beta = 0.5, 0.5
        return alpha * mse_loss + beta * ssim_loss
    '''
    '''
    def forward_loss(self, imgs, pred, mask):
        """
        Vectorized Dice loss on masked patches only.
        imgs: [N,1,H,W]
        pred: [N,L,P]   where P = patch_size**2
        mask: [N,L]     0=keep, 1=masked
        """
        # 1) sigmoid‐bound your raw logits into [0,1]
        pred = torch.sigmoid(pred)                          # [N, L, P]

        # 2) get targets
        target = self.patchify(imgs)                        # [N, L, P]
        # no norm_pix_loss for binary
        N, L, P = pred.shape

        # 3) flatten the patch dimension and batch+patch dims
        pred_flat   = pred.reshape(N*L, P)                  # [N*L, P]
        target_flat = target.reshape(N*L, P)                # [N*L, P]
        mask_flat   = mask.reshape(N*L)                     # [N*L]

        # 4) select only the masked patches
        sel = mask_flat.bool()
        pred_sel   = pred_flat[sel]                         # [M, P]
        target_sel = target_flat[sel]                       # [M, P]

        # 5) compute Dice in one shot
        #    intersection per‐patch
        inter = (pred_sel * target_sel).sum(dim=1)           # [M]
        #    sums per‐patch
        sums = pred_sel.sum(dim=1) + target_sel.sum(dim=1)   # [M]
        smooth = 1e-6
        dice_score = (2*inter + smooth) / (sums + smooth)   # [M]
        dice_loss  = 1.0 - dice_score                        # [M]

        # 6) mean over all masked patches
        return dice_loss.mean()
    '''
    '''
    def forward_hybrid_dice_bce_loss(self, imgs, pred, mask):
        """
        Hybrid Weighted BCEWithLogits + Dice loss on masked patches only.

        imgs:   [N, 1, H, W]      — ground‑truth images (0/1 binary)
        pred:   [N, L, P]         — raw decoder logits (no sigmoid yet)
        mask:   [N, L]            — 0 = keep (visible), 1 = remove (to predict)
        where L = number of patches, P = patch_size**2
        """
        # 1) Patchify the ground truth
        target = self.patchify(imgs)                  # [N, L, P]
        N, L, P = target.shape

        # 2) Flatten batch+patch dims
        pred_flat   = pred.reshape(N*L, P)               # [N*L, P]
        target_flat = target.reshape(N*L, P)             # [N*L, P]
        mask_flat   = mask.reshape(N*L)                  # [N*L]

        # 3) Select only the masked patches
        sel         = mask_flat.bool()                # [N*L]
        pred_sel    = pred_flat[sel]                  # [M, P]
        target_sel  = target_flat[sel]                # [M, P]

        # 4) Weighted BCEWithLogitsLoss
        #    Compute class balance on the selected patches
        pos = target_sel.sum()                        # # of white pixels
        neg = target_sel.numel() - pos                # # of black pixels
        eps = 1e-6
        pos_weight = (neg / (pos + eps)).clamp(min=1.0)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_sel, target_sel,
            pos_weight=pos_weight,
            reduction='mean'
        )

        # 5) Dice loss
        probs      = torch.sigmoid(pred_sel)          # [M, P]
        inter      = (probs * target_sel).sum(dim=1)  # [M]
        sums       = probs.sum(dim=1) + target_sel.sum(dim=1)  # [M]
        dice_score = (2*inter + eps) / (sums + eps)   # [M]
        dice_loss  = (1.0 - dice_score).mean()

        # 6) Combine
        alpha, beta = 0.5, 0.5
        return alpha * bce_loss + beta * dice_loss
    '''
    '''
    #weighted bce loss working okay 
    def forward_loss(self, imgs, pred, mask):
        # 1) patchify & flatten
        target = self.patchify(imgs)          # [N,L,P]
        N, L, P = target.shape
        pred_flat   = pred.reshape(N*L, P)
        target_flat = target.reshape(N*L, P)
        mask_flat   = mask.reshape(N*L)

        # 2) select masked patches
        sel         = mask_flat.bool()
        pred_sel    = pred_flat[sel]          # [M,P]
        target_sel  = target_flat[sel]        # [M,P]

        # 3) weighted BCE on those pixels
        #    (using dataset‑level pos_weight for stability)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_sel, target_sel,
            pos_weight=self.global_pos_weight,
            reduction='mean'
        )
        return bce_loss
    '''
    def forward_loss(self, imgs, pred, mask):
        """
        Hybrid MSE + BCEWithLogits on masked patches.

        imgs: [N,1,H,W]      ground‑truth binary images
        pred: [N,L,P]        raw decoder logits
        mask: [N,L]          1=masked, 0=visible
        """
        # 1) patchify ground truth
        target = self.patchify(imgs)           # [N, L, P]
        N, L, P = target.shape

        # 2) flatten and select masked patches
        pred_flat   = pred.reshape(N*L, P)
        target_flat = target.reshape(N*L, P)
        mask_flat   = mask.reshape(N*L).bool()

        pred_sel   = pred_flat[mask_flat]      # [M, P]
        target_sel = target_flat[mask_flat]    # [M, P]

        # 3) probabilities for MSE
        pred_prob = torch.sigmoid(pred_sel)    # [M, P]

        # 4) MSE loss (pixel‑wise L2)
        mse_loss = F.mse_loss(pred_prob, target_sel, reduction='mean')

        # 5) BCEWithLogits (on logits) with global pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_sel,
            target_sel,
            pos_weight=self.global_pos_weight,
            reduction='mean'
        )

        # 6) Combine
        alpha, beta = 0.4, 0.6
        return alpha * mse_loss + beta * bce_loss
        
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*1]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b_bw(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b_bw(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b_bw(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16_bw = mae_vit_base_patch16_dec512d8b_bw  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16_bw = mae_vit_large_patch16_dec512d8b_bw  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14_bw = mae_vit_huge_patch14_dec512d8b_bw  # decoder: 512 dim, 8 blocks 
