# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import Dict, Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat
from torch import nn

from fastvideo.models.stepvideo.modules.blocks import PatchEmbed, StepVideoTransformerBlock
from fastvideo.models.stepvideo.modules.normalization import AdaLayerNormSingle, PixArtAlphaTextProjection
from fastvideo.models.stepvideo.parallel import parallel_forward
from fastvideo.models.stepvideo.utils import with_empty_init

import os
import cv2
import numpy as np
def my_vis_attn(hidden_states, encoder_hidden_states, attention_mask, BCTHW, blk_idx, tokens, timestep_idx, meta):
    B, C, T, H, W = BCTHW
    video_embed = hidden_states[0].clone().float()
    text_actual_length = attention_mask[0, 0, video_embed.shape[0]:].sum()
    # 处理token文本
    tokens = [token[1:] if token.startswith("Ġ") else token for token in tokens]
    text_tokens = tokens[:text_actual_length]

    tokens_prompt = " ".join(text_tokens[1:])
    text_embed = encoder_hidden_states[0].clone().float()

    # 解填充处理
    text_unpadded_embed = text_embed[:text_actual_length, :] # [l_text, C]
    
    #  [l_text, T, H, W]
    d_k = text_unpadded_embed.shape[-1]
    atten_map = torch.matmul(text_unpadded_embed, video_embed.transpose(0, 1)) / np.sqrt(d_k)
    atten_map = atten_map.reshape(text_actual_length, T, H, W)
    # atten_map = torch.softmax(atten_map, dim=0)  # [l_text, T, H, W]
    # 输出目录
    output_dir = f"attention_maps/{tokens_prompt}"
    os.makedirs(output_dir, exist_ok=True)

    # --- 每个子图的目标尺寸 ---
    subplot_height, subplot_width = 720 // 8, 1280 // 8
    margin = 20  # 子图间距
    title_height = 40  # 主标题高度
    label_height = 30  # 为子图标签新增的高度
    colorbar_width = 30  # 颜色条宽度
    colorbar_margin = 80  # 颜色条与子图的间距
    colorbar_height = subplot_height

    frame_token_tensor_dict = {}
    for frame_idx in range(T):
        frame_attention = atten_map[:, frame_idx]  # [l_text, H, W]

        # --- 动态计算画布尺寸 ---
        n_tokens = len(text_tokens)
        max_cols = 5  # 每行最多8个子图
        n_cols = min(n_tokens, max_cols)
        n_rows = (n_tokens + n_cols - 1) // n_cols


        # 总画布尺寸
        canvas_width = n_cols * subplot_width + (n_cols + 1) * margin + colorbar_width + colorbar_margin
        canvas_height = title_height + n_rows * (subplot_height + label_height) + (n_rows + 1) * margin

        # 创建画布（白色背景）
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

        # --- 统一颜色尺度 ---
        vmin, vmax = frame_attention.min(), frame_attention.max()
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

        # --- 新增颜色条绘制 ---
        # 生成垂直渐变色条
        # 生成颜色条数据（正确的数值方向）
        colorbar_values = torch.linspace(vmax, vmin, colorbar_height).to(vmin.device)  # 从大到小！顶部=max
        normalized_values = ((colorbar_values - vmin) / (vmax - vmin) * 255).cpu().numpy().astype(np.uint8)

        # 应用颜色映射（不再需要反转！）
        colorbar = cv2.applyColorMap(normalized_values, cv2.COLORMAP_AUTUMN)
        colorbar = cv2.resize(colorbar, (colorbar_width, subplot_height))
        
        # 放置颜色条（右侧居中）
        cb_x = canvas_width - colorbar_width - colorbar_margin
        cb_y = title_height + (canvas_height - title_height - subplot_height) // 2
        canvas[cb_y:cb_y+subplot_height, cb_x:cb_x+colorbar_width] = colorbar
        
        # 添加数值标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        # 最大值标签（顶部）
        cv2.putText(canvas, f"{vmax:.2f}", 
                   (cb_x + colorbar_width + 5, cb_y + 20),
                   font, font_scale, (0,0,0), 1, cv2.LINE_AA)
        # 最小值标签（底部）
        cv2.putText(canvas, f"{vmin:.2f}", 
                   (cb_x + colorbar_width + 5, cb_y + subplot_height - 10),
                   font, font_scale, (0,0,0), 1, cv2.LINE_AA)

        token_tensor_dict = {}
        # --- 绘制每个token的子图 ---
        for i in range(n_tokens):
            row = i // n_cols
            col = i % n_cols
            
            # 计算子图位置（包含标签区域）
            x_start = margin + col * (subplot_width + margin)
            y_start = title_height + margin + row * (subplot_height + label_height + margin)

            # 处理当前token的热图
            import pdb
            pdb.set_trace()
            token_attention = frame_attention[i]  # [H, W]
            norm_attention = ((token_attention - vmin) / (vmax - vmin) * 255)
            norm_attention = norm_attention.cpu().numpy().astype(np.uint8)

            # 缩放热图
            resized_attention = cv2.resize(norm_attention, (subplot_width, subplot_height), 
                                         interpolation=cv2.INTER_CUBIC)
            colored_attention = cv2.applyColorMap(resized_attention, cv2.COLORMAP_AUTUMN)

            # 将热图粘贴到画布（放在标签下方）
            canvas[y_start+label_height:y_start+label_height+subplot_height, 
                   x_start:x_start+subplot_width] = colored_attention

            # --- 在子图上方添加token标签 ---
            text = text_tokens[i]
            token_tensor_dict[text] = resized_attention  # save token tensor

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6  # 比子图内标签稍大
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # 计算标签居中位置
            text_x = x_start + (subplot_width - text_size[0]) // 2
            text_y = y_start + label_height - 5  # 距离热图顶部5像素
            
            # 黑色文字（白色背景已保证可读性）
            cv2.putText(
                canvas, text,
                (text_x, text_y),
                font, font_scale, (0, 0, 0), thickness,
                lineType=cv2.LINE_AA
            )

        # --- 添加主标题 ---
        frame_token_tensor_dict[frame_idx] = token_tensor_dict
        # title = f"Frame {frame_idx} | Block {blk_idx} | Timestep {timestep_idx}"
        title = f"Timestep {timestep_idx} | Block {blk_idx} | Frame {frame_idx}"
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.8
        title_thickness = 2
        title_size = cv2.getTextSize(title, title_font, title_scale, title_thickness)[0]
        cv2.putText(
            canvas, title,
            ((canvas_width - title_size[0]) // 2, 30),
            title_font, title_scale, (0, 0, 0), title_thickness,
            lineType=cv2.LINE_AA
        )

        # --- 保存图像 ---
        save_path = os.path.join(output_dir, f"timestep_{timestep_idx}_{meta}_blk_{blk_idx}_frame_{frame_idx}.png")
        cv2.imwrite(save_path, canvas)
    # save npy array
    npy_save_dir = os.path.join("attention_maps_all", tokens_prompt)
    os.makedirs(npy_save_dir, exist_ok=True)
    npy_save_path = os.path.join(npy_save_dir, f"timestep_{timestep_idx}_{meta}_blk_{blk_idx}_frame_token_tensor.npy")
    np.save(npy_save_path, frame_token_tensor_dict)

class StepVideoModel(ModelMixin, ConfigMixin):
    _no_split_modules = ["StepVideoTransformerBlock", "PatchEmbed"]

    @with_empty_init
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 128,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 48,
        dropout: float = 0.0,
        patch_size: int = 1,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        use_additional_conditions: Optional[bool] = False,
        caption_channels = [6144, 1024],
        attention_type: Optional[str] = "parallel",
    ):
        super().__init__()

        # Set some common variables used across the board.
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels

        self.use_additional_conditions = use_additional_conditions

        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList([
            StepVideoTransformerBlock(dim=self.inner_dim,
                                      attention_head_dim=self.config.attention_head_dim,
                                      attention_type=attention_type) for _ in range(self.config.num_layers)
        ])

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)
        self.patch_size = patch_size

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, use_additional_conditions=self.use_additional_conditions)

        if isinstance(self.config.caption_channels, int):
            caption_channel = self.config.caption_channels
        else:
            caption_channel, clip_channel = self.config.caption_channels
            self.clip_projection = nn.Linear(clip_channel, self.inner_dim)

        self.caption_norm = nn.LayerNorm(caption_channel, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channel, hidden_size=self.inner_dim)

        self.parallel = attention_type == 'parallel'

    def patchfy(self, hidden_states):
        hidden_states = rearrange(hidden_states, 'b f c h w -> (b f) c h w')
        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        kv_seqlens = encoder_attention_mask.sum(dim=1).int()
        mask = torch.zeros([len(kv_seqlens), q_seqlen, max(kv_seqlens)],
                           dtype=torch.bool,
                           device=encoder_attention_mask.device)
        encoder_hidden_states = encoder_hidden_states[:, :max(kv_seqlens)]
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 1
        return encoder_hidden_states, mask

    @parallel_forward
    def block_forward(self,
                      hidden_states,
                      encoder_hidden_states=None,
                      timestep=None,
                      rope_positions=None,
                      attn_mask=None,
                      parallel=True,
                      mask_strategy=None):

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states,
                                  encoder_hidden_states,
                                  timestep=timestep,
                                  attn_mask=attn_mask,
                                  rope_positions=rope_positions,
                                  mask_strategy=mask_strategy[i])

        return hidden_states

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        fps: torch.Tensor = None,
        return_dict: bool = True,
        mask_strategy=None,
    ):
        assert hidden_states.ndim == 5
        "hidden_states's shape should be (bsz, f, ch, h ,w)"

        bsz, frame, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        hidden_states = self.patchfy(hidden_states)
        len_frame = hidden_states.shape[1]

        if self.use_additional_conditions:
            added_cond_kwargs = {
                "resolution": torch.tensor([(height, width)] * bsz,
                                           device=hidden_states.device,
                                           dtype=hidden_states.dtype),
                "nframe": torch.tensor([frame] * bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                "fps": fps
            }
        else:
            added_cond_kwargs = {}

        timestep, embedded_timestep = self.adaln_single(timestep, added_cond_kwargs=added_cond_kwargs)

        encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))

        if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
            clip_embedding = self.clip_projection(encoder_hidden_states_2)
            encoder_hidden_states = torch.cat([clip_embedding, encoder_hidden_states], dim=1)

        hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()
        encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask,
                                                                  encoder_hidden_states,
                                                                  q_seqlen=frame * len_frame)

        hidden_states = self.block_forward(hidden_states,
                                           encoder_hidden_states,
                                           timestep=timestep,
                                           rope_positions=[frame, height, width],
                                           attn_mask=attn_mask,
                                           parallel=self.parallel,
                                           mask_strategy=mask_strategy)

        hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)

        embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()

        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(shape=(-1, height, width, self.patch_size, self.patch_size,
                                                     self.out_channels))

        hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
        output = hidden_states.reshape(shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size))

        output = rearrange(output, '(b f) c h w -> b f c h w', f=frame)

        if return_dict:
            return {'x': output}
        return output
