# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenize Op used by Grain"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any
import grain.python as grain
import numpy as np
from MaxText import tokenizer


@dataclasses.dataclass
class TokenizeAndTrim(grain.MapTransform):
  """Tokenize and trim features to sequence length."""

  # pylint: disable=attribute-defined-outside-init
  text_column: str
  sequence_length: int
  add_bos: bool
  add_eos: bool
  tokenizer: tokenizer.SentencePieceTokenizerGrain | tokenizer.HFTokenizer

  def __post_init__(self):
    self._processor = None
    self._initialize_processor_lock = threading.Lock()

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Maps to each element."""
    if self._processor is None:
      with self._initialize_processor_lock:
        if self._processor is None:  # Ensures only one thread initializes SPP.
          self._processor = self.tokenizer

    text = element[self.text_column]
    token_ids = self._processor.encode(text)[:self.sequence_length]
    element[self.text_column] = np.asarray(token_ids, dtype=np.int32)
    return element

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_processor"]
    del state["_initialize_processor_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._processor = None
    self._initialize_processor_lock = threading.Lock()


@dataclasses.dataclass
class TokenizeAndChunk(grain.experimental.FlatMapTransform):
  """Tokenize and chunk features into multiple examples of sequence length."""

  # pylint: disable=attribute-defined-outside-init
  text_column: str
  sequence_length: int
  add_bos: bool
  add_eos: bool
  tokenizer: tokenizer.SentencePieceTokenizerGrain | tokenizer.HFTokenizer
  max_fan_out: int = 2048

  def __post_init__(self):
    self._processor = None
    self._initialize_processor_lock = threading.Lock()

  def flat_map(self, element: dict[str, Any]) -> list[dict[str, Any]]:
    """Maps one element to a LIST of chunked elements."""
    if self._processor is None:
      with self._initialize_processor_lock:
        if self._processor is None:  # Ensures only one thread initializes SPP.
          self._processor = self.tokenizer

    text = element.pop(self.text_column)
    max_len = self.sequence_length

    token_ids = self._processor.encode(text)

    if not token_ids:
      return []

    token_ids = np.array(token_ids, dtype=np.int32)

    output_elements = []

    for i in range(0, len(token_ids), max_len):
      new_element = {**element, self.text_column: token_ids[i : i + max_len]}
      output_elements.append(new_element)
    return output_elements

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_processor"]
    del state["_initialize_processor_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._processor = None
    self._initialize_processor_lock = threading.Lock()