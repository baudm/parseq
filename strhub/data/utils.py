# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.categorical import Categorical


class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = re.sub(self.unsupported, '', label)
        return label


class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _truncate(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._truncate(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs
            


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        self.specials_first = (self.EOS,)
        self.specials_last = (self.BOS, self.PAD)
        #? why was EOS first, and BOS PAD at the end?
        super().__init__(charset, self.specials_first, self.specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in self.specials_first + self.specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _truncate(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Truncate to [E]
        
        Returns:
            probs : list of probs up to [E]
            ids : list of ids before [E]
        
        """
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids
    
    def sample(self, b_logits: Tensor, greedy=False, temp=1.0, max_label_length=25, pad_to_max_length=True, device=None):
        batch_tokens = []
        for logits in b_logits:
            #- get eos positions
            dist = logits.softmax(-1)
            probs, ids = dist.max(-1)  # greedy selection
            probs, ids = self._truncate(probs, ids)
            if greedy:
                tokens = self._ids2tok(ids, True)
                batch_tokens.append(tokens)
            else:
                eos_idx = min((len(ids), max_label_length))
                #- sample
                logits_temp  = logits / temp
                # exclude special characters
                logits_temp = logits_temp[:, len(self.specials_first):-len(self.specials_last)]
                dist_temp = logits_temp.softmax(-1)
                m = Categorical(dist_temp)
                sampled_ids = m.sample()[:eos_idx]
                sampled_ids += len(self.specials_first)
                tokens = self._ids2tok(sampled_ids, True)
                batch_tokens.append(tokens)
        #- encode
        seq = self.encode(batch_tokens, device=device)
        seq = seq[:, :-1] # remove eos
        if pad_to_max_length:
            seq = F.pad(seq, (0, max_label_length + 1 - seq.shape[1]), "constant", self.pad_id)
        return seq
