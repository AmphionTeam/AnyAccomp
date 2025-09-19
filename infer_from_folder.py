import torch
import json
from anyaccomp.inference_utils import Sing2SongInferencePipeline
import os
import random
import librosa
import numpy as np
import soundfile as sf
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="AnyAccomp Inference Script from Folder"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./config/flow_matching.json",
        help="Path to the configuration file for the model",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./pretrained/flow_matching",
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--vocoder_checkpoint_path",
        type=str,
        default="./pretrained/vocoder",
        help="Path to the vocoder checkpoint file",
    )
    parser.add_argument(
        "--vocoder_cfg_path",
        type=str,
        default="./config/vocoder.json",
        help="Path to the vocoder configuration file",
    )
    parser.add_argument(
        "--infer_dst",
        type=str,
        default="./example/output",
        help="Destination directory for inference results",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./example/input",
        help="Path to the source folder containing vocal audio files",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=50,
        help="Number of timesteps for reverse diffusion",
    )
    parser.add_argument(
        "--cfg", type=float, default=3, help="CFG scale for reverse diffusion"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to run inference on (e.g., "cuda" or "cpu")',
    )
    parser.add_argument(
        "--seed", type=int, default=1024, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set fixed seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    cfg_path = args.cfg_path
    checkpoint_path = args.checkpoint_path
    vocoder_checkpoint_path = args.vocoder_checkpoint_path
    vocoder_cfg_path = args.vocoder_cfg_path
    input_folder = args.input_folder
    n_timesteps = args.n_timesteps
    cfg = args.cfg
    device = args.device
    infer_dst = args.infer_dst
    os.makedirs(infer_dst, exist_ok=True)

    with open(os.path.join(infer_dst, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    acc_dst = os.path.join(infer_dst, "accompaniment")
    mixture_dst = os.path.join(infer_dst, "mixture")
    os.makedirs(acc_dst, exist_ok=True)
    os.makedirs(mixture_dst, exist_ok=True)

    # Initialize pipeline
    inference_pipeline = Sing2SongInferencePipeline(
        checkpoint_path,
        cfg_path,
        vocoder_checkpoint_path,
        vocoder_cfg_path,
        device=device,
    )

    inference_pipeline.sample_rate = 24000

    supported_extensions = (".wav", ".mp3", ".flac")
    audio_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(supported_extensions)
    ]

    print(f"Found {len(audio_files)} audio files in {input_folder}")

    for vocal_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            vocal_audio, _ = librosa.load(vocal_path, sr=24000, mono=True)
            vocal_tensor = torch.tensor(vocal_audio).unsqueeze(0).to(device)
            vocal_mel = inference_pipeline.encode_vocal(vocal_tensor)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                mel = inference_pipeline.model.reverse_diffusion(
                    vocal_mel=vocal_mel,
                    n_timesteps=n_timesteps,
                    cfg=cfg,
                )

            mel = mel.float()
            wav = inference_pipeline._generate_audio(mel)  # [1, T]
            wav = wav.squeeze().detach().cpu().numpy()
            wav = librosa.util.fix_length(data=wav, size=len(vocal_audio))
            mixture_wav = wav + vocal_audio
            base_filename = os.path.basename(vocal_path)
            sf.write(os.path.join(acc_dst, base_filename), wav, 24000)
            sf.write(os.path.join(mixture_dst, base_filename), mixture_wav, 24000)

        except Exception as e:
            print(f"Error processing file {vocal_path}: {e}")
            continue


if __name__ == "__main__":
    main()
