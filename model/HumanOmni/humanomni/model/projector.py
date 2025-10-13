#    Copyright 2024 Alibaba DAMO Academy
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
from transformers import TRANSFORMERS_CACHE


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'mm_projector.bin')):
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)



def build_vision_projector(config, delay_load=False, projector_type=None, **kwargs):
    if projector_type is None:
        projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "stc_connector":
        return STCConnector(config)
    elif projector_type == "stp_connector":
        return STPConnector(config)
    elif projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    elif projector_type == "stc_connector_v35_half":
        return STCConnectorV35_half(config)
    elif projector_type == "stc_connector_our":
        return STCConnectorOur(config)
    elif projector_type == "spatial_conv":
        return SpatialConv(config)
    elif projector_type == "spatial_pool":
        return SpatialPool(config)
    if projector_type == 'identity':
        return IdentityMap()
    if projector_type == 'stc_connector_v35_mlp' or projector_type == 'stc_connector_v35_or_mlp' :
        return STCConnectorWithMLP(config)
    if projector_type == 'stc_connector_our_mlp_bert' :
        return STCConnectorOurWithMLP(config)
    if projector_type == 'all_in_one' :
        return AllInOne(config)
    if projector_type == 'all_in_one_mlp' :
        return AllInOneMLP(config)
    if projector_type == 'all_in_one_small' :
        return AllInOneSmall(config)
    if projector_type == 'pllava' :
        return PllavaMultiModalProjector(config)
    raise ValueError(f'Unknown projector type: {projector_type}')

def build_audio_projector(config, delay_load=False, **kwargs):
    if hasattr(config, 'mm_audio_projector_type'):
        projector_type = getattr(config, 'mm_audio_projector_type')
    else:
        projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.audio_hidden_size, config.hidden_size)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.audio_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    if projector_type == 'identity':
        return IdentityMap()
    raise ValueError(f'Unknown projector type: {projector_type}')

def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class PllavaMultiModalProjector(nn.Module):
    supported_highres = ['pad_crop_four', 'slide', ]
    def __init__(self, config):
        super().__init__()
        # 复用 STCConnector 的配置参数
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        
        # PLLAVA 特有的配置参数，写死或使用默认值
        self.use_pooling = True
        self.frame_shape = (27, 27)  # 默认值
        self.num_frames = getattr(config, 'num_frames', 16)  # 如果config中没有，使用默认值16
        self.pooling_shape = (8, 14, 14)  # 固定值
        
        # 模型层定义
        self.pooling = nn.AdaptiveAvgPool3d(self.pooling_shape)
        self.linear_1 = nn.Linear(encoder_hidden_size, hidden_size, bias=True)
        self.act = nn.SiLU()  # 使用与 STCConnector 相同的激活函数
        self.linear_2 = nn.Linear(hidden_size, output_hidden_size, bias=True)


    def convert_Fembeddings2video(self, input, num_videos, frame_shape):
        input = einops.rearrange(input, 
                                '(num_videos num_frames) (h w) embed_dims -> num_videos embed_dims num_frames h w', 
                                num_videos=num_videos, h=frame_shape[0])
        return input
    
    def convert_video2Fembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> (num_videos num_frames) (h w) embed_dims ', )
        return input

    def convert_video2MMembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> num_videos (num_frames h w) embed_dims ', )
        return input

    def forward(self, x, split_sizes=None, num_frames=None):
        """
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
            split_sizes: 每个样本的帧数列表，如果是图片则为1
            num_frames: 目标帧数
        Returns:
            aggregated tokens [b, l, d]
        """
        if split_sizes is not None:
            # 处理混合输入（图片和视频）
            x = torch.split(x, split_sizes)
            features = []
            for idx, feat in enumerate(x):
                if feat.shape[0] == 1:  # 图片输入
                    feat = feat.expand(num_frames, -1, -1)
                    features.append(feat)
                else:  # 视频输入
                    features.append(feat)
            x = torch.stack(features, dim=0)
            b, t, l, d = x.shape
        else:
            # 单一类型输入
            if x.ndim == 5:  # [b, t, h, w, d]
                b, t, h, w, d = x.shape
                x = einops.rearrange(x, 'b t h w d -> b t (h w) d')
            else:  # [b, t, l, d]
                b, t, l, d = x.shape

        # 确保空间维度正确
        if l != self.frame_shape[0] * self.frame_shape[1]:
            hw = int(l ** 0.5)
            assert hw * hw == l, f"Input spatial dimension {l} is not a perfect square"
            self.frame_shape = (hw, hw)

        # 线性投影
        hidden_states = self.linear_1(x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        # 转换为视频格式
        hidden_states = einops.rearrange(hidden_states, 'b t l d -> (b t) l d')
        hidden_states_videos = self.convert_Fembeddings2video(hidden_states, b, self.frame_shape)
        
        # 池化
        hidden_states_videos = self.pooling(hidden_states_videos)
        
        # 重塑输出格式
        hidden_states = einops.rearrange(hidden_states_videos, 'b d t h w -> b (t h w) d')
        
        return hidden_states

class AllInOne(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        if depth != 0:
            self.s1_video = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
            self.s1_body = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1_video = nn.Identity()
            self.s1_body = nn.Identity()

        self.sampler_video = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=(2, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=True
            ),
            nn.SiLU()
        )
        self.sampler_body = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=(2, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2_video = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
            self.s2_body = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2_video = nn.Identity()
            self.s2_body = nn.Identity()
        self.readout_video = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        self.readout_body = build_mlp(mlp_depth, hidden_size, output_hidden_size)

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp_2xgelu_face = nn.Sequential(*modules)

    def forward(self, video, body, face):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        # import ipdb;ipdb.set_trace()
        t = video.size(1)

        if video.ndim == 4:
            hw = int(video.size(2) ** 0.5)
            x = einops.rearrange(video, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif video.ndim == 5:
            x = einops.rearrange(video, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1_video(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler_video(x)
        x = x[:, :, :-1, :, :]
      #  print(x.shape)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2_video(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        video_feature = self.readout_video(x)

        if body.ndim == 4:
            hw = int(body.size(2) ** 0.5)
            x = einops.rearrange(body, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif body.ndim == 5:
            x = einops.rearrange(body, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1_body(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler_body(x)
        x = x[:, :, :-1, :, :]
      #  print(x.shape)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2_body(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        body_feature = self.readout_body(x)

        mlp_res = self.mlp_2xgelu_face(face)
        return video_feature, body_feature, mlp_res


class AllInOneSmall(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=2, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        if depth != 0:
            self.s1_video = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
            self.s1_body = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1_video = nn.Identity()
            self.s1_body = nn.Identity()

        self.sampler_video = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=(2, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
                bias=True
            ),
            nn.SiLU()
        )
        self.sampler_body = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=(2, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2_video = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
            self.s2_body = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2_video = nn.Identity()
            self.s2_body = nn.Identity()
        self.readout_video = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        self.readout_body = build_mlp(mlp_depth, hidden_size, output_hidden_size)

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp_2xgelu_face = nn.Sequential(*modules)

    def forward(self, video, body, face):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
       # import ipdb;ipdb.set_trace()
        t = video.size(1)

        if video.ndim == 4:
            hw = int(video.size(2) ** 0.5)
            x = einops.rearrange(video, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif video.ndim == 5:
            x = einops.rearrange(video, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1_video(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler_video(x)
        x = x[:, :, :-1, :, :]
      #  print(x.shape)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2_video(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        video_feature = self.readout_video(x)

        if body.ndim == 4:
            hw = int(body.size(2) ** 0.5)
            x = einops.rearrange(body, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif body.ndim == 5:
            x = einops.rearrange(body, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1_body(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler_body(x)
        x = x[:, :, :-1, :, :]
      #  print(x.shape)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2_body(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        body_feature = self.readout_body(x)

        mlp_res = self.mlp_2xgelu_face(face)
        return video_feature, body_feature, mlp_res

class AllInOneMLP(nn.Module):

    def __init__(self, config):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp_2xgelu_face = nn.Sequential(*modules)

        modules_body = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules_body.append(nn.GELU())
            modules_body.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp_2xgelu_body = nn.Sequential(*modules_body)

        modules_ori = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules_ori.append(nn.GELU())
            modules_ori.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp_2xgelu_ori = nn.Sequential(*modules_ori)

    def forward(self, video, body, face):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        # import ipdb;ipdb.set_trace()
        mlp_ori = self.mlp_2xgelu_ori(video)
        mlp_face = self.mlp_2xgelu_face(face)
        mlp_body = self.mlp_2xgelu_body(body)
        return mlp_ori, mlp_body, mlp_face


class STCConnectorOurWithMLP(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()

        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=(2, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp_2xgelu = nn.Sequential(*modules)

    def forward(self, x, split_sizes, num_frames):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        # import ipdb;ipdb.set_trace()
        x_origin = x
        x = torch.split(x, split_sizes)
        image_feats = []
        for idx, image_feat in enumerate(x):
            if image_feat.shape[0] == 1:
                image_feat = image_feat.expand(num_frames, -1, -1)
                image_feats.append(image_feat)
            else:
                image_feats.append(image_feat)
        x = torch.stack(image_feats, dim=0)
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        x = x[:, :, :-1, :, :]
      #  print(x.shape)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)

        mlp_res = self.mlp_2xgelu(x_origin)
        return x, mlp_res

class STCConnectorWithMLP(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU())
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp_2xgelu = nn.Sequential(*modules)

    def forward(self, x, split_sizes, num_frames):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        # import ipdb;ipdb.set_trace()
        x_origin = x
        x = torch.split(x, split_sizes)
        image_feats = []
        for idx, image_feat in enumerate(x):
            if image_feat.shape[0] == 1:
                image_feat = image_feat.expand(num_frames, -1, -1)
                image_feats.append(image_feat)
            else:
                image_feats.append(image_feat)
        x = torch.stack(image_feats, dim=0)
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)

        mlp_res = self.mlp_2xgelu(x_origin)
        return x, mlp_res

class STCConnectorOur(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
       # self.downsample = downsample
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()

        ### # our_3  ####
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=(2, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
     #   print("$$$$", x.shape)
        x = x[:, :, :-1, :, :]
     #   print("#####", x.shape)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
     #   print("!!!!", x.shape)
        return x


class STCConnector(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
      #  import ipdb;ipdb.set_trace()
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        return x

class STPConnector(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())


class STCConnectorV35(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU())

class STCConnectorV35_half(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=1, mlp_depth=1):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU())


class SpatialConv(STCConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


class SpatialPool(STPConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
