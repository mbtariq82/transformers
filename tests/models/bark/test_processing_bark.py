# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import unittest

import numpy as np

from transformers import AutoTokenizer, BarkProcessor
from transformers.testing_utils import require_torch, slow


AudioInput = Union[
    str,  # wav path, or base64 string
    np.ndarray,  # 1-D float array
    List[str],
    List[np.ndarray],
]


@require_torch
class BarkProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "suno/bark-small"
        self.tmpdirname = tempfile.mkdtemp()
        self.voice_preset = "en_speaker_1"
        self.input_string = "This is a test string"
        self.speaker_embeddings_dict_path = "speaker_embeddings_path.json"
        self.speaker_embeddings_directory = "speaker_embeddings"

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer)

        processor.save_pretrained(self.tmpdirname)
        processor = BarkProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())

    @slow
    def test_save_load_pretrained_additional_features(self):
        processor = BarkProcessor.from_pretrained(
            pretrained_processor_name_or_path=self.checkpoint,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
        )

        # TODO (ebezzam) not all speaker embedding are properly downloaded.
        # My hypothesis: there are many files (~700 speaker embeddings) and some fail to download (not the same at different first runs)
        # https://github.com/huggingface/transformers/blob/967045082faaaaf3d653bfe665080fd746b2bb60/src/transformers/models/bark/processing_bark.py#L89
        # https://github.com/huggingface/transformers/blob/967045082faaaaf3d653bfe665080fd746b2bb60/src/transformers/models/bark/processing_bark.py#L188
        # So for testing purposes, we will remove the unavailable speaker embeddings before saving.
        processor._verify_speaker_embeddings(remove_unavailable=True)
        processor.save_pretrained(
            self.tmpdirname,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
            speaker_embeddings_directory=self.speaker_embeddings_directory,
        )

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")

        processor = BarkProcessor.from_pretrained(
            self.tmpdirname,
            self.speaker_embeddings_dict_path,
            bos_token="(BOS)",
            eos_token="(EOS)",
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

    def test_speaker_embeddings(self):
        processor = BarkProcessor.from_pretrained(
            pretrained_processor_name_or_path=self.checkpoint,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
        )

        seq_len = 35
        nb_codebooks_coarse = 2
        nb_codebooks_total = 8

        voice_preset = {
            "semantic_prompt": np.ones(seq_len),
            "coarse_prompt": np.ones((nb_codebooks_coarse, seq_len)),
            "fine_prompt": np.ones((nb_codebooks_total, seq_len)),
        }

        # test providing already loaded voice_preset
        inputs = processor(text=self.input_string, voice_preset=voice_preset)

        processed_voice_preset = inputs["history_prompt"]
        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())

        # test loading voice preset from npz file
        tmpfilename = os.path.join(self.tmpdirname, "file.npz")
        np.savez(tmpfilename, **voice_preset)
        inputs = processor(text=self.input_string, voice_preset=tmpfilename)
        processed_voice_preset = inputs["history_prompt"]

        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())

        # test loading voice preset from the hub
        inputs = processor(text=self.input_string, voice_preset=self.voice_preset)

    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer)

        encoded_processor = processor(text=self.input_string)

        encoded_tok = tokenizer(
            self.input_string,
            padding="max_length",
            max_length=256,
            add_special_tokens=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key].squeeze().tolist())





    def test_encoder_decoder(self):
        ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav"

        ### this should be in modeling ###
        qwen_tokenizer = Qwen3TTSTokenizer.from_pretrained(self.model_id)   
        qwen_encoded_frames = qwen_tokenizer.encode(ref_audio_path_1)
        print("Qwen Encoded:", qwen_encoded_frames, qwen_encoded_frames.audio_codes[0].shape)
        #wavs1, out_sr1 = qwen_tokenizer.decode(qwen_encoded_frames)
        #print("Decoded audio:", wavs1)  # getting NaN values
       # sf.write("decoded_single_12hz.wav", wavs1[0], out_sr1)
        ###

        ### this should be in modeling ###
        hf_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(self.model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
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
        audio_codes_padded = hf_encoded_frames.audio_codes[0].unsqueeze(0).to(device)
        hf_decoded_frames = hf_tokenizer.decode(audio_codes_padded)
        print("HF Decoded audio:", hf_decoded_frames)
        wav_tensors = hf_decoded_frames.audio_values
        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        out_sr1 = hf_tokenizer.get_output_sample_rate()
        #sf.write("hf_decoded_single_12hz_hf.wav", wavs[0], out_sr1)

        assert torch.allclose(torch.tensor(qwen_encoded_frames.audio_codes[0]), hf_encoded_frames.audio_codes[0], atol=1e-5), "Encoded audio codes do not match!"
        print("Encoded audio codes match!")