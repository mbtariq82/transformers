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
from transformers.models.qwen3_tts.modeling_qwen3_tts import Qwen3TTSTokenizerV2Model
from transformers import AutoFeatureExtractor, AutoProcessor

#@require_torch?
class Qwen3TTSProcessorTest(unittest.TestCase):
    processor_class = Qwen3TTSTokenizerV2Model
    model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    def test_save_load_pretrained(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        self.assertIsNotNone(processor.feature_extractor)
        self.assertIsNotNone(processor.tokenizer)

    def test_audio_processing(self):
        processor = AutoProcessor.from_pretrained(self.model_id)

        

        self.assertIn("input_values", inputs)
        self.assertEqual(inputs["input_values"].dtype, torch.float32)
        #self.assertEqual(






    def test_tokenizer(self):
        tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(self.model_id)
        
