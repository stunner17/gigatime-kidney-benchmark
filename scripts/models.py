"""GigaTIME, GigaTIME-Flash and Prov-GigaPath: architectures and loaders.

GigaTIME (UNet++) and GigaTIME-Flash (DINOv2-small + LoRA encoder, conv decoder) are copied unchanged in behaviour from
the official repository, github.com/prov-gigatime/GigaTIME (scripts/archs.py, scripts/gigatime_flash_testing.ipynb).
Weights are gated on Hugging Face; request access and log in (huggingface_hub.login) before the first run.
"""
import os
import numpy as np
import torch
from torch import nn

REPOS = {'gigatime': 'prov-gigatime/GigaTIME', 'gigatime_flash': 'prov-gigatime/GigaTIME-flash',
         'gigapath': 'prov-gigapath/prov-gigapath'}
# GigaTIME output channels, in order (TRITC and Cy5 are background channels)
CHANNELS = ['DAPI', 'TRITC', 'Cy5', 'PD-1', 'CD14', 'CD4', 'T-bet', 'CD34', 'CD68', 'CD16', 'CD11c', 'CD138', 'CD20',
            'CD3', 'CD8', 'PD-L1', 'CK', 'Ki67', 'Tryptase', 'Actin-D', 'Caspase3-D', 'PHH3-B', 'Transgelin']
MEAN = np.array([0.485, 0.456, 0.406], np.float32)           # ImageNet normalisation, used by all three models
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ GigaTIME (UNet++), from scripts/archs.py
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class GigaTIME(nn.Module):
    def __init__(self, num_classes=23, input_channels=3):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = VGGBlock(input_channels, f[0], f[0])
        self.conv1_0 = VGGBlock(f[0], f[1], f[1])
        self.conv2_0 = VGGBlock(f[1], f[2], f[2])
        self.conv3_0 = VGGBlock(f[2], f[3], f[3])
        self.conv4_0 = VGGBlock(f[3], f[4], f[4])
        self.conv0_1 = VGGBlock(f[0] + f[1], f[0], f[0])
        self.conv1_1 = VGGBlock(f[1] + f[2], f[1], f[1])
        self.conv2_1 = VGGBlock(f[2] + f[3], f[2], f[2])
        self.conv3_1 = VGGBlock(f[3] + f[4], f[3], f[3])
        self.conv0_2 = VGGBlock(f[0] * 2 + f[1], f[0], f[0])
        self.conv1_2 = VGGBlock(f[1] * 2 + f[2], f[1], f[1])
        self.conv2_2 = VGGBlock(f[2] * 2 + f[3], f[2], f[2])
        self.conv0_3 = VGGBlock(f[0] * 3 + f[1], f[0], f[0])
        self.conv1_3 = VGGBlock(f[1] * 3 + f[2], f[1], f[1])
        self.conv0_4 = VGGBlock(f[0] * 4 + f[1], f[0], f[0])
        self.final = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return self.final(x0_4)


# ------------------------------------------------------------------ GigaTIME-flash, from gigatime_flash_testing.ipynb
class LoraConfig:
    def __init__(self, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
        self.r, self.lora_alpha, self.lora_dropout = r, lora_alpha, lora_dropout
        self.target_modules = target_modules or []


class LoRALinear(nn.Module):
    def __init__(self, base_layer, config):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = {"default": config.lora_alpha / config.r}
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(config.lora_dropout)})
        self.lora_A = nn.ModuleDict({"default": nn.Linear(base_layer.in_features, config.r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(config.r, base_layer.out_features, bias=False)})

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        update = self.lora_B["default"](self.lora_A["default"](self.lora_dropout["default"](x)))
        return self.base_layer(x) + update * self.scaling["default"]


class BaseModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class PeftFallbackModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = BaseModelWrapper(model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model.model, name)

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


def apply_lora(module, config):
    for child_name, child in list(module.named_children()):
        if child_name in config.target_modules and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, config))
        else:
            apply_lora(child, config)


def make_vit_small_patch16_dinov2():
    from timm.layers import SwiGLUPacked
    from timm.models.vision_transformer import _create_vision_transformer
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, init_values=1e-5,
                      mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, img_size=224)
    return _create_vision_transformer("vit_small_patch14_dinov2", pretrained=False, **model_args)


class GigaTIMEFlash(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        vit_encoder = make_vit_small_patch16_dinov2()
        apply_lora(vit_encoder, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "proj"]))
        self.encoder = PeftFallbackModel(vit_encoder)
        self.num_classes = num_classes
        self.decoder4 = self._decoder_block(384, 192)
        self.decoder3 = self._decoder_block(192, 96)
        self.decoder2 = self._decoder_block(96, 48)
        self.decoder1 = self._decoder_block(48, 24)
        self.skip1 = self._skip_block(384, 48, 3)
        self.skip2 = self._skip_block(384, 96, 2)
        self.skip3 = self._skip_block(384, 192, 1)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        self.encoder.patch_embed.img_size = (256, 256)
        self.encoder.patch_embed.grid_size = (16, 16)
        self.encoder.patch_embed.num_patches = 256

    @staticmethod
    def _decoder_block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1),
                             nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))

    @staticmethod
    def _skip_block(i, o, times):
        layers = []
        for _ in range(times):
            layers.append(nn.ConvTranspose2d(i, o, kernel_size=2, stride=2)); i = o
        return nn.Sequential(*layers)

    @staticmethod
    def _tokens_to_patch(x):
        b, n, c = x.shape
        g = int(n ** 0.5)
        return x.permute(0, 2, 1).contiguous().view(b, c, g, g)

    @staticmethod
    def _resize_pos_embed(pos_embed, new_grid_size):
        cls_tok, grid_tok = pos_embed[:, :1], pos_embed[:, 1:]
        old = int(grid_tok.shape[1] ** 0.5)
        grid_tok = grid_tok.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        grid_tok = nn.functional.interpolate(grid_tok, size=new_grid_size, mode="bicubic", align_corners=False)
        grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)
        return torch.cat((cls_tok, grid_tok), dim=1)

    def forward(self, x):
        enc_outs = []
        x = self.encoder.patch_embed(x)
        g = int(x.shape[1] ** 0.5)
        x = x + self._resize_pos_embed(self.encoder.pos_embed, (g, g))[:, 1:]
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)
        for idx, block in enumerate(self.encoder.blocks):
            x = block(x)
            if idx in {3, 5, 8, 11}:
                enc_outs.append(x)
        x = self.decoder4(self._tokens_to_patch(enc_outs[3]))
        x = self.decoder3(x + self.skip3(self._tokens_to_patch(enc_outs[2])))
        x = self.decoder2(x + self.skip2(self._tokens_to_patch(enc_outs[1])))
        x = self.decoder1(x + self.skip1(self._tokens_to_patch(enc_outs[0])))
        x = self.final_conv(x)
        if x.shape[-1] != 256:
            x = nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)
        return x


# ------------------------------------------------------------------ loaders
def _weights(repo):
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo_id=repo)
    return os.path.join(d, 'model.pth')


def load_gigatime(repo, device):
    m = GigaTIME(23, 3)
    sd = torch.load(_weights(repo), map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    m.load_state_dict(sd)                                   # strict: every tensor must match
    return m.to(device).eval(), dict(matched=len(sd), expected=len(m.state_dict()))


def load_gigatime_flash(repo, device):
    """same key remapping as the official notebook; reports how many tensors matched (must be all)"""
    m = GigaTIMEFlash(23)
    ck = torch.load(_weights(repo), map_location='cpu')
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    ms = m.state_dict(); loaded = {}
    for k, v in ck.items():
        cands = [k]
        if k.startswith('module.'):
            cands.append(k[7:])
        if '.base_layer.' in k:
            cands.append(k.replace('.base_layer.', '.'))
        if k.startswith('encoder.') and not k.startswith('encoder.base_model.model.'):
            cands.append(k.replace('encoder.', 'encoder.base_model.model.', 1))
        for c in cands:
            if c in ms and ms[c].shape == v.shape:
                loaded[c] = v; break
    m.load_state_dict(loaded, strict=False)
    return m.to(device).eval(), dict(matched=len(loaded), expected=len(ms))


def load_gigapath(repo, device):
    import timm
    m = timm.create_model(f'hf_hub:{repo}', pretrained=True)
    return m.to(device).eval(), dict(params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 1))


def to_tensor(rgb):
    """(N, H, W, 3) uint8 -> normalised float tensor (N, 3, H, W)"""
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


@torch.no_grad()
def virtual_mif(model, rgb, window, device, batch=16, amp=False):
    """sigmoid probability maps (23, H, W) for an RGB image whose sides are multiples of `window`;
    non-overlapping windows as in the official inference code. amp=False keeps fp32 like the official code."""
    H, W, _ = rgb.shape
    tiles, pos = [], []
    for i in range(0, H, window):
        for j in range(0, W, window):
            tiles.append(rgb[i:i + window, j:j + window]); pos.append((i, j))
    out = np.zeros((23, H, W), np.float32)
    for s in range(0, len(tiles), batch):
        x = to_tensor(np.stack(tiles[s:s + batch])).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            y = model(x)
        y = torch.sigmoid(y.float()).cpu().numpy()
        for k, (i, j) in enumerate(pos[s:s + batch]):
            out[:, i:i + window, j:j + window] = y[k]
    return out
