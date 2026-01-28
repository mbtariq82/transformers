import torch
import requests
import soundfile as sf
import io
import torchaudio
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSTokenizerV2EncoderOutput,
    Qwen3TTSTokenizerV2Encoder,
)
from qwen_tts import Qwen3TTSTokenizer      # original repo


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    encoder_valid_num_quantizers = 16 # value copied from Qwen3TTSTokenizerV2Config
    encode_downsample_rate = 1920


    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map=None,
        dtype=torch.float32,
        attn_implementation="eager"
    )








    
    audio = tokenizer.load_audio(ref_audio_path_1, target_sr=24000)
    audio = torch.from_numpy(audio) #?
    input_values = audio.unsqueeze(0)  # (1, C, T)
    padding_mask = torch.ones_like(input_values, dtype=torch.bool)

    ### encode audio using top level class from qwen3-tts ###
    audios = {
        "input_values": input_values,
        "padding_mask": padding_mask,
    }
    qwen_encoded_frames = tokenizer.encode(audios)
    #upstream_codes = encoded_frames.audio_codes[:, :encoder_valid_num_quantizers]
    #upstream_codes = [code[..., :-(-mask.sum() // encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]


    ### using ported tokenizer encoder ###
    encoder = Qwen3TTSTokenizerV2Encoder.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=None,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    hf_encoded_frames = encoder.encode(input_values=input_values, return_dict=True)
    #audio_codes = encoded_frames.audio_codes[:, :encoder_valid_num_quantizers]
    #hf_codes = [code[..., :-(-mask.sum() // encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]
    #print("Input:", input_values.shape)
    #print("Encoded type:", type(encoded_frames))
    #print("audio_codes shape:", encoded_frames.audio_codes.shape)
    #print("dtype:", encoded_frames.audio_codes.dtype)
    #print("min/max:", encoded_frames.audio_codes.min(), encoded_frames.audio_codes.max())
    
    print(hf_encoded_frames, qwen_encoded_frames)
    print(torch.equal(hf_encoded_frames, qwen_encoded_frames))

if __name__ == "__main__":
    main()