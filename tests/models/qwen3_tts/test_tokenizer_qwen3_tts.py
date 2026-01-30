import torch
import requests
import soundfile as sf
import io
import torchaudio
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers.models.mimi.configuration_mimi import MimiConfig
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSTokenizerV2EncoderOutput,
    Qwen3TTSTokenizerV2Encoder,
)
from qwen_tts import Qwen3TTSTokenizer      # original repo

def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav"
    tokenizer_model_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_name,
        device_map=None,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    audio = tokenizer.load_audio(ref_audio_path_1, target_sr=24000)
    input_values = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    qwen_encoded_frames = tokenizer.encode(audio, sr=24000)
    print("Qwen Encoded:", qwen_encoded_frames, qwen_encoded_frames.audio_codes[0].shape)

    ### only port what is needed to recreate the above ###
    # https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/blob/main/model.safetensors
    #Qwen3TTSTokenizerV2Model
    #├── encoder
    #│   ├── downsample
    #│   ├── encoder
    #│   ├── encoder_transformer
    #│   └── quantizer
    #└── decoder
    # worst case scenario is porting everything up to Qwen3TTSTokenizerV2Model in order to run .encoder.encode
    # model = Qwen3TTSTokenizerV2Model.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")





    # load Qwen3TTSTokenizerV2Encoder with the encoder weights only
    state_dict = load_file(
        hf_hub_download(
            repo_id=tokenizer_model_name,
            filename="model.safetensors"
        )
    )
    encoder_state_dict = {
        k[len("encoder."):]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    config = MimiConfig.from_pretrained(tokenizer_model_name)
    encoder = Qwen3TTSTokenizerV2Encoder(config)
    missing, unexpected = encoder.load_state_dict(
        encoder_state_dict,
        strict=True
    )
    print("Missing:", missing)
    print("Unexpected:", unexpected)
    hf_encoded_frames = encoder.encode(input_values=input_values, return_dict=True)
    print("HF Encoded:", hf_encoded_frames, hf_encoded_frames.audio_codes[0].shape)



if __name__ == "__main__":
    main()