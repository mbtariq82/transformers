import unittest
import time
import torch
import soundfile as sf
import numpy as np
import io
import urllib.request
import librosa
from typing import List, Optional, Union
from urllib.parse import urlparse
from transformers.models.qwen3_tts.modeling_qwen3_tts import Qwen3TTSTokenizerV2Model, Qwen3TTSTokenizerV2Config
from transformers import AutoFeatureExtractor, AutoProcessor

AudioInput = Union[
    str,  # wav path, or base64 string
    np.ndarray,  # 1-D float array
    List[str],
    List[np.ndarray],
]

class Qwen3TTSTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cls._get_test_config()
        #cls.model = Qwen3TTSTokenizerV2Model(cls.config).eval()
        cls.model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        cls.expected_encoding = torch.from_numpy(
            np.load("expected_encoder_output.npy")
        )
        cls.expected_decoding = torch.from_numpy(
            np.load("expected_decoder_output.npy")
        )
        cls.tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(cls.model_id)
        cls.feature_extractor = AutoFeatureExtractor.from_pretrained(cls.model_id)



    @staticmethod
    def _get_test_config():
        return Qwen3TTSTokenizerV2Config(
            input_sample_rate=16000,
            output_sample_rate=16000,
            encode_downsample_rate=320,
            decode_upsample_rate=320,
            encoder_valid_num_quantizers=4,
            encoder_config={
                "hidden_size": 64,
                "num_layers": 2,
            },
            decoder_config={
                "hidden_size": 64,
                "num_layers": 2,
            },
        )

    def test_save_and_load_pretrained(self):
        pass # TODO

    def test_tokenizer_model_init_from_config(self):
        pass # TODO
    #    model = Qwen3TTSTokenizerV2Model(self.config)
#
    #    self.assertEqual(
    #        model.get_input_sample_rate(),
    #        self.config.input_sample_rate,
    #    )
    #    self.assertEqual(
    #        model.get_output_sample_rate(),
    #        self.config.output_sample_rate,
    #    )
    #    self.assertEqual(
    #        model.get_encode_downsample_rate(),
    #        self.config.encode_downsample_rate,
    #    )
    #    self.assertEqual(
    #        model.get_decode_upsample_rate(),
    #        self.config.decode_upsample_rate,
    #    )

    def test_encoder(self):
        target_sr = int(self.feature_extractor.sampling_rate)
        inputs = self.feature_extractor(
            raw_audio=self.expected_decoding,
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        device = getattr(self.tokenizer, "device", None)
        inputs = inputs.to(device).to(self.tokenizer.dtype)
        encoded_frames = self.tokenizer.encode(inputs["input_values"].squeeze(1), inputs["padding_mask"].squeeze(1)).audio_codes[0].cpu()
        assert encoded_frames.shape == self.expected_encoding.shape, (
            f"Shape mismatch: {encoded_frames.shape} vs {self.expected_encoding.shape}"
        )
        assert torch.equal(encoded_frames, self.expected_encoding), (
            "Encoded audio codes do not match hardcoded reference!"
        )

    def test_decoder(self):
        decoded_frames = self.tokenizer.decode(self.expected_encoding.unsqueeze(0))
        wav = decoded_frames.audio_values[0]  # torch.Tensor

        assert wav.shape == torch.Size([252885]), (
            f"Shape mismatch: {wav.shape} vs {torch.Size([252885])}"
        )

        assert torch.allclose(
            wav.cpu(),
            self.expected_decoding,
            atol=1e-6,
        ), (
            f"Decoded wavs do not match hardcoded reference:\n"
            f"{wav} vs {self.expected_decoding}"
        )



