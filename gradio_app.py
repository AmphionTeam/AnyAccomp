import torch
import json
from anyaccomp.inference_utils import Sing2SongInferencePipeline
import os
import random
import librosa
import numpy as np
import soundfile as sf
import gradio as gr
import time

base_dir = os.path.dirname(
    os.path.abspath(__file__)
)

CFG_PATH = os.path.join(base_dir, "./config/flow_matching.json")
CHECKPOINT_PATH = os.path.join(
    base_dir, "./pretrained/flow_matching"
)
VOCODER_CHECKPOINT_PATH = os.path.join(
    base_dir, "./pretrained/vocoder"
)
VOCODER_CFG_PATH = os.path.join(base_dir, "./config/vocoder.json")
INFER_DST = os.path.join(base_dir, "./example/output_gradio")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(INFER_DST, exist_ok=True)
acc_dst = os.path.join(INFER_DST, "accompaniment")
mixture_dst = os.path.join(INFER_DST, "mixture")
os.makedirs(acc_dst, exist_ok=True)
os.makedirs(mixture_dst, exist_ok=True)

print("Initializing AnyAccomp InferencePipeline...")
try:
    inference_pipeline = Sing2SongInferencePipeline(
        CHECKPOINT_PATH,
        CFG_PATH,
        VOCODER_CHECKPOINT_PATH,
        VOCODER_CFG_PATH,
        device=DEVICE,
    )
    inference_pipeline.sample_rate = 24000
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    inference_pipeline = None


def sing2song_inference(vocal_filepath, n_timesteps, cfg_scale, seed):
    if inference_pipeline is None:
        raise gr.Error(
            "Model could not be loaded. Please check paths and environment configuration."
        )

    if vocal_filepath is None:
        raise gr.Error("Please upload a vocal audio file.")

    if seed == -1 or seed is None:
        seed = random.randint(0, 2**32 - 1)

    seed = int(seed)
    print(f"Using seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    try:
        duration = librosa.get_duration(path=vocal_filepath)
        if not (3 <= duration <= 30):
            raise gr.Error("Audio duration must be between 3 and 30 seconds.")
    except Exception as e:
        raise gr.Error(f"Cannot read audio file or get duration: {e}")

    try:
        vocal_audio, _ = librosa.load(vocal_filepath, sr=24000, mono=True)
        vocal_tensor = torch.tensor(vocal_audio).unsqueeze(0).to(DEVICE)

        vocal_mel = inference_pipeline.encode_vocal(vocal_tensor)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            mel = inference_pipeline.model.reverse_diffusion(
                vocal_mel=vocal_mel,
                n_timesteps=int(n_timesteps),
                cfg=cfg_scale,
            )

        mel = mel.float()
        wav = inference_pipeline._generate_audio(mel)
        wav = wav.squeeze().detach().cpu().numpy()

        wav = librosa.util.fix_length(data=wav, size=len(vocal_audio))
        mixture_wav = wav + vocal_audio

        timestamp = int(time.time())
        original_filename = os.path.basename(vocal_filepath)
        base_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp}.wav"

        accompaniment_path = os.path.join(acc_dst, base_filename)
        mixture_path = os.path.join(mixture_dst, base_filename)

        sf.write(accompaniment_path, wav, 24000)
        sf.write(mixture_path, mixture_wav, 24000)

        return accompaniment_path, mixture_path, "Status: Complete!"

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise gr.Error(f"An error occurred during processing: {e}")


def randomize_seed():
    return random.randint(0, 2**32 - 1)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AnyAccomp: GENERALIZABLE ACCOMPANIMENT GENERATION
        Upload a 3-30 second vocal or instrument track (.wav or .mp3) and the model will generate an accompaniment for it.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload or Select Audio")
            vocal_input = gr.Audio(
                type="filepath",
                label="Upload Vocal or Instrument Audio",
                sources=["upload", "microphone"],
            )

            example1_path = os.path.join(
                base_dir, "./example/gradio/example1.mp3"
            )
            example2_path = os.path.join(
                base_dir, "./example/gradio/example2.wav"
            )
            example3_path = os.path.join(
                base_dir, "./example/gradio/example3.wav"
            )
            gr.Examples(
                examples=[[example1_path], [example2_path], [example3_path]],
                inputs=[vocal_input],
                label="Or click an example to start",
            )
            gr.Markdown("### 2. Adjust Parameters (Optional)")
            with gr.Accordion("Advanced Settings", open=True):
                n_timesteps_slider = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Inference Steps (n_timesteps)",
                )
                cfg_slider = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="CFG Scale"
                )

                with gr.Row():
                    seed_input = gr.Number(
                        value=-1, label="Seed (-1 for random)", precision=0
                    )
                    random_seed_btn = gr.Button("🎲")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Generate and Listen to the Result")

            status_text = gr.Markdown("Status: Ready")

            accompaniment_output = gr.Audio(
                label="Generated Accompaniment", type="filepath"
            )
            mixture_output = gr.Audio(
                label="Mixture (Vocal + Accompaniment)", type="filepath"
            )

            submit_btn = gr.Button("Generate Accompaniment", variant="primary")

    submit_btn.click(
        fn=sing2song_inference,
        inputs=[vocal_input, n_timesteps_slider, cfg_slider, seed_input],
        # The function will now update the status text as its third output
        outputs=[accompaniment_output, mixture_output, status_text],
    )

    random_seed_btn.click(fn=randomize_seed, inputs=None, outputs=seed_input)

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8091)
