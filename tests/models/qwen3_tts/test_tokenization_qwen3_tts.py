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
class Qwen3TTSTokenizerTest(unittest.TestCase):
    tokenizer = Qwen3TTSTokenizerV2Model
    model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    