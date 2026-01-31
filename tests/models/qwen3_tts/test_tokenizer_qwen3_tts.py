import unittest
import torch
import soundfile as sf
import numpy as np
import io
import urllib.request
import librosa
from typing import List, Optional, Union
from urllib.parse import urlparse
from transformers.models.qwen3_tts.modeling_qwen3_tts import Qwen3TTSTokenizerV2Model
from transformers import AutoFeatureExtractor
from qwen_tts import Qwen3TTSTokenizer      # original repo

AudioInput = Union[
    str,  # wav path, or base64 string
    np.ndarray,  # 1-D float array
    List[str],
    List[np.ndarray],
]

#@require_torch
class Qwen3TTSTokenizerIntegrationTest(unittest.TestCase):
    pretrained_model_name_or_path = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        # Heuristic: no filesystem path separators and long enough.
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False
        
    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def load_audio(self, x: str, target_sr: int) -> np.ndarray:
        """
        Load audio from wav path or base64 string, then resample to target_sr.

        Args:
            x (str):
                A wav file path, or a base64 audio string (raw or data URL).
            target_sr (int):
                Target sampling rate.

        Returns:
            np.ndarray:
                1-D float32 waveform at target_sr.
        """
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

        return audio.astype(np.float32)

    def normalize_audio_inputs(self, audios: AudioInput, sr: Optional[int], target_sr: int = 24000) -> List[np.ndarray]:
            """
            Normalize all supported input types into a list of 1-D numpy float32 waveforms
            at `self.feature_extractor.sampling_rate`.

            Args:
                audios (AudioInput):
                    - str: wav path OR base64 audio string
                    - np.ndarray: raw waveform (sr must be provided)
                    - list[str] / list[np.ndarray]
                sr (Optional[int]):
                    Sampling rate for raw numpy input. Required if input is np.ndarray or list[np.ndarray].

            Returns:
                List[np.ndarray]:
                    List of float32 waveforms resampled to model input SR.
            """
            if isinstance(audios, (str, np.ndarray)):
                audios = [audios]

            if len(audios) == 0:
                return []

            if isinstance(audios[0], str):
                # wav path list or base64 list
                return [self.load_audio(x, target_sr=target_sr) for x in audios]  # type: ignore[arg-type]

            # numpy list
            if sr is None:
                raise ValueError("For numpy waveform input, you must provide `sr` (original sampling rate).")

            out: List[np.ndarray] = []
            for a in audios:  # type: ignore[assignment]
                if not isinstance(a, np.ndarray):
                    raise TypeError("Mixed input types are not supported. Use all paths/base64 or all numpy arrays.")
                if a.ndim > 1:
                    a = np.mean(a, axis=-1)
                if int(sr) != target_sr:
                    a = librosa.resample(y=a.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
                out.append(a.astype(np.float32))
            return out

    def test_encoder_decoder(self):
        ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_2.wav"
        tokenizer_model_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


        qwen_tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_model_name)
        
        qwen_encoded_frames = qwen_tokenizer.encode(ref_audio_path_1)
        print("Qwen Encoded:", qwen_encoded_frames, qwen_encoded_frames.audio_codes[0].shape)
        
        #wavs1, out_sr1 = tokenizer.decode(qwen_encoded_frames)
        #print("Decoded audio:", wavs1)  # getting NaN values
        #sf.write("tests/models/qwen3_tts/decoded_single_12hz.wav", wavs1[0], out_sr1)


        hf_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_model_name)
        
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.pretrained_model_name_or_path)
        target_sr = int(feature_extractor.sampling_rate)

        wavs = self.normalize_audio_inputs(ref_audio_path_1, sr=24000, target_sr=target_sr)
        inputs = feature_extractor(
            raw_audio=wavs,
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        device = getattr(hf_tokenizer, "device", None)
        inputs = inputs.to(device).to(hf_tokenizer.dtype)

        hf_encoded_frames = hf_tokenizer.encode(inputs["input_values"].squeeze(1), inputs["padding_mask"].squeeze(1))
        print("HF Encoded:", hf_encoded_frames, hf_encoded_frames.audio_codes[0].shape)

        wavs1, out_sr1 = hf_tokenizer.decode(hf_encoded_frames.audio_codes[0].unsqueeze(0))
        print("HF Decoded audio:", wavs1)  
        sf.write("tests/models/qwen3_tts/hf_decoded_single_12hz_hf.wav", wavs1[0], out_sr1)


        assert torch.allclose(torch.tensor(qwen_encoded_frames.audio_codes[0]), hf_encoded_frames.audio_codes[0], atol=1e-5), "Encoded audio codes do not match!"
        print("Encoded audio codes match!")