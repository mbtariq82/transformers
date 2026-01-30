import torch
import soundfile as sf
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers.models.mimi.configuration_mimi import MimiConfig
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSTokenizerV2Encoder,
)
from qwen_tts import Qwen3TTSTokenizer      # original repo

def main():
    #torch.manual_seed(0)
    #torch.use_deterministic_algorithms(True)
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_2.wav"
    tokenizer_model_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_name,
        device_map=None,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    qwen_encoded_frames = tokenizer.encode(ref_audio_path_1)
    print("Qwen Encoded:", qwen_encoded_frames, qwen_encoded_frames.audio_codes[0].shape)
    wavs1, out_sr1 = tokenizer.decode(qwen_encoded_frames)
    print("Decoded audio:", wavs1)  # getting NaN values
    sf.write("transformers/tests/models/qwen3_tts/decoded_single_12hz.wav", wavs1[0], out_sr1)


    audio = tokenizer.load_audio(ref_audio_path_1, target_sr=24000)
    input_values = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    # TODO: Qwen3TTSTokenizer.load_audio

    # Qwen3TTSTokenizer.encode
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
    encoder.load_state_dict(
        encoder_state_dict,
        strict=True
    )
    hf_encoded_frames = encoder.encode(input_values=input_values, return_dict=True) # currently using MimiModel.encode
    print("HF Encoded:", hf_encoded_frames, hf_encoded_frames.audio_codes[0].shape)

    # TODO: HF Decoded






    # model = Qwen3TTSTokenizerV2Model.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")



if __name__ == "__main__":
    main()