import math
import json
import librosa
import torch
import torchaudio
import accelerate
import safetensors
import numpy as np
import os
import yaml

import torchvision
from librosa.feature import chroma_stft

import torchvision
import random
import numpy as np

import sys


from anyaccomp.fmt_model import FlowMatchingTransformerConcat
from models.codec.amphion_codec.vocos import Vocos
from models.codec.melvqgan.melspec import MelSpectrogram
from models.codec.coco.rep_coco_model import CocoContentStyle, CocoContent, CocoStyle

from tqdm import tqdm

from utils.util import load_config

import io

from transformers import T5Tokenizer, T5EncoderModel

import warnings


class Sing2SongInferencePipeline:
    def __init__(
        self,
        checkpoint_path,
        cfg_path,
        vocoder_checkpoint_path,
        vocoder_cfg_path,
        device="cuda",
    ):
        self.cfg = load_config(cfg_path)
        self.device = device

        self.checkpoint_path = checkpoint_path
        self._load_model(checkpoint_path)

        self._build_input_model()
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.vocoder_cfg = load_config(vocoder_cfg_path)
        self._build_output_model()
        print("Output model built")

    def _load_model(self, checkpoint_path):
        self.model = FlowMatchingTransformerConcat(
            cfg=self.cfg.model.flow_matching_transformer
        )

        accelerate.load_checkpoint_and_dispatch(self.model, checkpoint_path)
        self.model.eval().to(self.device)
        print(
            f"model Params: {round(sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6, 2)}M"
        )
        print(f"Loaded model from {checkpoint_path}")

    def _build_input_model(self):
        self.coco_model = CocoStyle(
            cfg=self.cfg.model.coco, construct_only_for_quantizer=True
        )
        self.coco_model.eval()
        self.coco_model.to(self.device)
        accelerate.load_checkpoint_and_dispatch(
            self.coco_model, self.cfg.model.coco.pretrained_path
        )

    def _build_output_model(self):
        # print(vocoder_checkpoint_path)
        self.vocoder = Vocos(cfg=self.vocoder_cfg.model.vocos)
        accelerate.load_checkpoint_and_dispatch(
            self.vocoder, self.vocoder_checkpoint_path
        )
        self.vocoder = self.vocoder.eval().to(self.device)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def _extract_coco_codec(self, speech):
        """
        Args:
            speech: [B, T]
        Returns:
            codecs: [B, T]. Note that codecs might be not at 50Hz!
        """
        target_chroma_dim = self.cfg.model.coco.chromagram_dim

        speech = speech.cpu().numpy().squeeze()

        chromagram = chroma_stft(
            y=speech,
            sr=self.cfg.preprocess.chromagram.sample_rate,
            n_fft=self.cfg.preprocess.chromagram.n_fft,
            hop_length=self.cfg.preprocess.chromagram.hop_size,
            win_length=self.cfg.preprocess.chromagram.win_size,
            n_chroma=target_chroma_dim,
        ).T  # [D, T] -> [T, D]
        chromagram_feats = torch.tensor(chromagram).unsqueeze(0).to(self.device)
        codecs, _ = self.coco_model.quantize(chromagram_feats)
        return codecs

    @torch.no_grad()
    def encode_vocal(self, speech):  # (B, T)
        speech = speech.to(self.device)
        codecs = self._extract_coco_codec(speech)
        return codecs

    @torch.no_grad()
    def _generate_audio(self, mel):
        synthesized_audio = (self.vocoder(mel.transpose(1, 2)).detach().cpu())[0]

        return synthesized_audio
