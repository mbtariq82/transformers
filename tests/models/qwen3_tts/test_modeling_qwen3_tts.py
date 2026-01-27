import torch
import requests
import soundfile as sf
import io
import torchaudio
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSTokenizerV2EncoderOutput,
    Qwen3TTSTokenizerV2Encoder,
)
from transformers.models.mimi.configuration_mimi import MimiConfig
from qwen3_tts import Qwen3TTSTokenizer      # original repo


if __name__=="__main__":
    main()

def load_audio_from_url(url: str, target_sr: int = 24000):
    # Download
    response = requests.get(url)
    response.raise_for_status()

    # Decode audio
    audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")

    # Shape → (channels, length)
    if audio.ndim == 1:
        audio = audio[None, :]   # (1, T)
    else:
        audio = audio.T          # (C, T)

    audio = torch.from_numpy(audio)

    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    return audio  # (channels, length)

def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    encoder.eval()

    encoder = Qwen3TTSTokenizerV2Encoder(config=MimiConfig())#**encoder_config)
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    audio = load_audio_from_url(ref_audio_path_1, target_sr=24000)
    # Add batch dimension
    input_values = audio.unsqueeze(0)
    # shape: (1, channels, length)    

    padding_mask = torch.ones_like(input_values, dtype=torch.bool)
    encoder_valid_num_quantizers = 16 # from Qwen3TTSTokenizerV2Config
    encode_downsample_rate = 1920

    encoded_frames = encoder.encode(input_values=input_values, return_dict=True)


    audio_codes = encoded_frames.audio_codes[:, :encoder_valid_num_quantizers]
    audio_codes = [code[..., :-(-mask.sum() // encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]
    
    print("Input:", input_values.shape)
    print("Encoded type:", type(encoded_frames))
    print("audio_codes shape:", encoded_frames.audio_codes.shape)
    print("dtype:", encoded_frames.audio_codes.dtype)
    print("min/max:", encoded_frames.audio_codes.min(), encoded_frames.audio_codes.max())


    hf_codes = audio_codes.cpu()
    
    model = Qwen3TTSTokenizer()#.from_pretrained(
        #"Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        #device_map=None,
        #dtype=torch.float32,
        #attn_implementation="eager"
    #)
    audio = model.load_audio(ref_audio_path_1, target_sr=24000)

    audio = torch.from_numpy(audio)

    input_values = audio.unsqueeze(0)  # (1, C, T)

    padding_mask = torch.ones(
        (input_values.shape[0], input_values.shape[-1]),
        dtype=torch.bool,
        device=input_values.device,
    )

    audios = {
        "input_values": input_values,
        "padding_mask": padding_mask,
    }
    encoded_frames = model.encode(audios)

    upstream_codes = encoded_frames.audio_codes[:, :encoder_valid_num_quantizers]
    upstream_codes = [code[..., :-(-mask.sum() // encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]

    print(torch.equal(hf_codes, upstream_codes))
    print((hf_codes - upstream_codes).abs().max())

if __name__ == "__main__":
    main()