import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from anyaccomp.llama_nar import DiffLlamaConcat
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast


class FlowMatchingTransformerConcat(nn.Module):
    def __init__(
        self,
        vocab_size=1024,
        mel_dim=100,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        cfg_scale=0.2,
        use_cond_code=False,
        cond_codebook_size=1024,
        cond_dim=1024,
        cond_scale_factor=1,
        sigma=1e-5,
        time_scheduler="linear",
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg

        mel_dim = (
            cfg.mel_dim if cfg is not None and hasattr(cfg, "mel_dim") else mel_dim
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        num_layers = (
            cfg.num_layers
            if cfg is not None and hasattr(cfg, "num_layers")
            else num_layers
        )
        num_heads = (
            cfg.num_heads
            if cfg is not None and hasattr(cfg, "num_heads")
            else num_heads
        )
        cfg_scale = (
            cfg.cfg_scale
            if cfg is not None and hasattr(cfg, "cfg_scale")
            else cfg_scale
        )
        use_cond_code = (
            cfg.use_cond_code
            if cfg is not None and hasattr(cfg, "use_cond_code")
            else use_cond_code
        )
        cond_codebook_size = (
            cfg.cond_codebook_size
            if cfg is not None and hasattr(cfg, "cond_codebook_size")
            else cond_codebook_size
        )
        cond_dim = (
            cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        )
        time_scheduler = (
            cfg.time_scheduler
            if cfg is not None and hasattr(cfg, "time_scheduler")
            else time_scheduler
        )
        sigma = cfg.sigma if cfg is not None and hasattr(cfg, "sigma") else sigma
        cond_scale_factor = (
            cfg.cond_scale_factor
            if cfg is not None and hasattr(cfg, "cond_scale_factor")
            else cond_scale_factor
        )

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.use_cond_code = use_cond_code
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim
        self.time_scheduler = time_scheduler
        self.sigma = sigma
        self.cond_scale_factor = cond_scale_factor

        self.vocab_size = (
            cfg.vocab_size
            if cfg is not None and hasattr(cfg, "vocab_size")
            else vocab_size
        )
        self.vocal_mel_proj = (
            nn.Linear(self.cfg.cond_code_dim, self.hidden_size)
            if not self.use_cond_code
            else nn.Sequential(
                nn.Embedding(
                    self.vocab_size, self.mel_dim
                ),  # [batch] -> [batch, mel_dim]
                nn.Linear(
                    self.mel_dim, self.hidden_size
                ),  # [batch, mel_dim] -> [batch, hidden_size]
            )
        )

        self.diff_estimator = DiffLlamaConcat(
            mel_dim=self.mel_dim,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            flash_attention=hasattr(cfg, "flash_attention") and cfg.flash_attention,
        )

        if hasattr(cfg, "repa_loss") and cfg.repa_loss.enable:
            repa_dim = (
                cfg.repa_loss.repa_dim
                if hasattr(cfg.repa_loss, "repa_dim")
                else self.hidden_size
            )
            self.repa_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, repa_dim),
            )

        self.reset_parameters()

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    @torch.no_grad()
    def forward_diffusion(self, x, t):
        """
        x: (B, T, mel_dim)
        t: (B,)
        """

        new_t = t
        t = t.unsqueeze(-1).unsqueeze(-1)
        z = torch.randn(
            x.shape, dtype=x.dtype, device=x.device, requires_grad=False
        )  # (B, T, mel_dim)

        cfg_scale = self.cfg_scale

        # get prompt len
        if torch.rand(1) > 0.7:
            prompt_len = torch.randint(
                min(x.shape[1] // 4, 5), int(x.shape[1] * 0.4), (x.shape[0],)
            ).to(
                x.device
            )  # (B,)
        else:
            prompt_len = torch.zeros(x.shape[0]).to(x.device)

        split_ratio = torch.rand(prompt_len.shape, device=prompt_len.device)  # (B,)

        left_len = (split_ratio * (prompt_len + 1).float()).long()  # (B,)
        right_len = prompt_len - left_len  # (B,)

        T = x.shape[1]
        is_prompt = torch.zeros_like(x[:, :, 0])  # (B, T)
        col_indices = torch.arange(T, device=x.device).repeat(x.shape[0], 1)  # (B, T)
        left_mask = col_indices < left_len.unsqueeze(1)
        right_mask = col_indices >= (T - right_len.unsqueeze(1))
        is_prompt[left_mask | right_mask] = 1

        mask = torch.ones_like(x[:, :, 0])  # mask if 1, not mask if 0
        mask[is_prompt.bool()] = 0
        mask = mask[:, :, None]

        # flow matching: xt = (1 - (1 - sigma) * t) * x0 + t * x; where x0 ~ N(0, 1), x is a sample
        # flow gt: x - (1 - sigma) * x0 = x - (1 - sigma) * noise
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, new_t, prompt_len, mask

    def loss_t(
        self,
        x,
        x_mask,
        t,
        lyric=None,
        output_hidden_states=False,
    ):
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(x, t)

        noise = z

        prompt_len = prompt_len.float()

        # drop condition using cfg_scale
        if lyric is not None:
            cfg_mask = torch.where(
                torch.rand_like(prompt_len) > self.cfg_scale,
                torch.ones_like(prompt_len),  # keep cond
                torch.zeros_like(prompt_len),  # drop cond
            ).to(lyric.device)

            cond_mask = cfg_mask[:, None, None]  # [b, 1, 1]

            lyric = lyric * cond_mask

        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        output = self.diff_estimator(
            xt, new_t, x_mask, lyric, output_hidden_states=output_hidden_states
        )
        if output_hidden_states:
            return_list = [noise, x, output["hidden_states"], final_mask, prompt_len]
            return_list.append(output["all_hidden_states"])
        else:
            return_list = [noise, x, output, final_mask, prompt_len]

        return return_list

    def compute_loss(self, x, x_mask, lyric=None, output_hidden_states=False):
        # x0: (B, T, num_quantizer)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        # from CosyVoice: considering the generation process at the beginning is harder than follows, we involve a cosine scheduler for the timestep t
        if self.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)
        else:
            pass
        return self.loss_t(
            x, x_mask, t, lyric, output_hidden_states=output_hidden_states
        )

    def forward(self, x, x_mask, vocal_mel, output_hidden_states=False):
        cond = self.vocal_mel_proj(vocal_mel)
        return self.compute_loss(x, x_mask, cond, output_hidden_states)

    @torch.no_grad()
    def reverse_diffusion(
        self,
        vocal_mel=None,
        prompt=None,
        right_prompt=None,
        x_mask=None,
        prompt_mask=None,
        right_prompt_mask=None,
        target_len=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        h = 1.0 / n_timesteps
        prompt_len = prompt.shape[1] if prompt is not None else 0
        right_prompt_len = right_prompt.shape[1] if right_prompt is not None else 0
        # print(prompt_len, right_prompt_len)
        if vocal_mel is not None:
            target_len = vocal_mel.shape[1]
        elif target_len is None:
            target_len = 1000  # hardcode 50Hz 20s
        else:
            raise ValueError
        full_len = target_len
        target_len = target_len - prompt_len - right_prompt_len

        cond = self.vocal_mel_proj(vocal_mel)

        if x_mask is None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)
        if prompt_mask is None and prompt is not None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(cond.device)
        if right_prompt_mask is None and right_prompt is not None:
            right_prompt_mask = torch.ones(cond.shape[0], right_prompt_len).to(
                cond.device
            )

        if prompt is not None and right_prompt is not None:
            xt_mask = torch.cat([prompt_mask, x_mask, right_prompt_mask], dim=1)
        elif prompt is not None and right_prompt is None:
            xt_mask = torch.cat([prompt_mask, x_mask], dim=1)
        elif prompt is None and right_prompt is not None:
            xt_mask = torch.cat([x_mask, right_prompt_mask], dim=1)
        else:
            xt_mask = x_mask

        z = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=cond.dtype,
            device=cond.device,
            requires_grad=False,
        )
        xt = z
        # t from 0 to 1: x0 = z ~ N(0, 1)
        for i in range(n_timesteps):
            if prompt is not None and right_prompt is not None:
                xt_input = torch.cat([prompt, xt, right_prompt], dim=1)
            elif prompt is not None and right_prompt is None:
                xt_input = torch.cat([prompt, xt], dim=1)
            elif prompt is None and right_prompt is not None:
                xt_input = torch.cat([xt, right_prompt], dim=1)
            else:
                xt_input = xt
            t = (0 + (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            flow_pred = self.diff_estimator(xt_input, t, xt_mask, cond)
            flow_pred = flow_pred[:, prompt_len : prompt_len + target_len, :]
            # cfg

            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(
                    xt_input, t, xt_mask, torch.zeros_like(cond)
                )
                uncond_flow_pred = uncond_flow_pred[
                    :, prompt_len : prompt_len + target_len, :
                ]
                pos_flow_pred_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescale_flow_pred = (
                    flow_pred_cfg * pos_flow_pred_std / flow_pred_cfg.std()
                )
                flow_pred = (
                    rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg
                )

            dxt = flow_pred * h
            xt = xt + dxt

        return xt
