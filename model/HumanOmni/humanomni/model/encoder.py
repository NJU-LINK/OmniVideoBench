import os

import torch
import torch.nn as nn

from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig,
     WhisperFeatureExtractor, WhisperProcessor, WhisperConfig, WhisperForAudioClassification
)

class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, raw_datas=None):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size

class WhisperAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_vision_select_layer
       
        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_audio_tower", False):
            # TODO: better detector is needed.
            print(f"The checkpoint seems to contain `audio_tower` weights: `unfreeze_mm_audio_tower`: True.")
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.audio_tower_name)
    def load_model(self, device_map=None):
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.audio_tower_name))
            return
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(self.audio_tower_name)
        self.audio_tower = WhisperForAudioClassification.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False)
        self.is_loaded = True
    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.hidden_states[self.select_layer]
        return audio_features
    def forward(self, samples):
        if isinstance(samples, list):
            audio_features = []
            for sample in samples:
            
                audio_forward_outs = self.audio_tower.encoder(sample, output_hidden_states=True)
                # audio_feature = self.feature_select(audio_forward_outs).to(audio_features.dtype)
                audio_features.append(audio_forward_outs.last_hidden_state)
        else:
            
            audio_forward_outs = self.audio_tower.encoder(samples, return_dict=True)
            audio_features = audio_forward_outs.last_hidden_state
        return audio_features
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    @property
    def dtype(self):
        return self.audio_tower.dtype
    @property
    def device(self):
        return self.audio_tower.device
    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only
    @property
    def hidden_size(self):
        return self.config.hidden_size


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if  'clip' in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower:
        vision_tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

    return vision_tower

def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    
    if "whisper" in audio_tower:
        return WhisperAudioTower(audio_tower, args=audio_tower_cfg, **kwargs)
