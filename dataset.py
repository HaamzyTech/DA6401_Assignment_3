from collections import Counter
import os
from typing import Iterable, List, Sequence, Tuple

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def _load_multi30k_split(split: str):
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    kwargs = {"split": split}
    if token:
        kwargs["token"] = token
    return load_dataset("bentrevett/multi30k", **kwargs)


class SimpleVocab:
    """Small vocabulary helper with the torchtext-like methods used here."""

    def __init__(
        self,
        counter: Counter,
        specials: Sequence[str] = SPECIALS,
        min_freq: int = 1,
    ) -> None:
        self.itos: List[str] = list(specials)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if freq < min_freq or token in self.stoi:
                continue
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

    def lookup_token(self, index: int) -> str:
        if index < 0 or index >= len(self.itos):
            return "<unk>"
        return self.itos[index]

    def lookup_indices(self, tokens: Iterable[str]) -> List[int]:
        return [self[token] for token in tokens]


class Multi30kDataset:
    _cached_vocabs = None

    def __init__(self, split="train"):
        """
        Load the Multi30k dataset and prepare German/English tokenized examples.
        Source language is German (de); target language is English (en).
        """
        self.split = self._normalize_split(split)
        self.src_tokenizer = spacy.blank("de")
        self.tgt_tokenizer = spacy.blank("en")
        self.raw_data = _load_multi30k_split(self.split)

        if Multi30kDataset._cached_vocabs is None:
            self.build_vocab()
        else:
            self.src_vocab, self.tgt_vocab = Multi30kDataset._cached_vocabs

        self.data = self.process_data()

    @staticmethod
    def _normalize_split(split: str) -> str:
        aliases = {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
        return aliases.get(split, split)

    @staticmethod
    def _get_text(example, lang: str) -> str:
        if lang in example:
            return example[lang]
        if "translation" in example and lang in example["translation"]:
            return example["translation"][lang]
        raise KeyError(f"Could not find language field '{lang}' in dataset example.")

    def _tokenize_src(self, text: str) -> List[str]:
        return [token.text.lower() for token in self.src_tokenizer(text.strip())]

    def _tokenize_tgt(self, text: str) -> List[str]:
        return [token.text.lower() for token in self.tgt_tokenizer(text.strip())]

    def build_vocab(self):
        """
        Build source and target vocabularies from the train split only.
        Validation/test examples are intentionally excluded to avoid leakage.
        """
        train_data = _load_multi30k_split("train")
        src_counter = Counter()
        tgt_counter = Counter()

        for example in train_data:
            src_counter.update(self._tokenize_src(self._get_text(example, "de")))
            tgt_counter.update(self._tokenize_tgt(self._get_text(example, "en")))

        self.src_vocab = SimpleVocab(src_counter, specials=SPECIALS, min_freq=1)
        self.tgt_vocab = SimpleVocab(tgt_counter, specials=SPECIALS, min_freq=1)
        Multi30kDataset._cached_vocabs = (self.src_vocab, self.tgt_vocab)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Convert German and English sentences into integer token tensors.
        Each sequence includes <sos> and <eos>.
        """
        processed = []
        for example in self.raw_data:
            src_tokens = self._tokenize_src(self._get_text(example, "de"))
            tgt_tokens = self._tokenize_tgt(self._get_text(example, "en"))

            src_ids = [SOS_IDX] + self.src_vocab.lookup_indices(src_tokens) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + self.tgt_vocab.lookup_indices(tgt_tokens) + [EOS_IDX]

            processed.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )
        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    @staticmethod
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
        return src_padded, tgt_padded
