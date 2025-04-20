from __future__ import annotations

import string
import unicodedata
from os import listdir
from os.path import isfile, join
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ALLOWED_CHARACTERS = string.ascii_letters + " .,;'" + "_"
N_CHARS = len(ALLOWED_CHARACTERS)


def unicode_to_ascii(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALLOWED_CHARACTERS
    )


def char_to_index(char: str) -> int:
    if len(char) > 1:
        raise ValueError("Given `char` has length greater than 1.")

    if char not in ALLOWED_CHARACTERS:
        char = "_"
    return ALLOWED_CHARACTERS.find(char)


def line_to_2d_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), N_CHARS)
    for i, char in enumerate(line):
        tensor[i, char_to_index(char)] = 1

    return tensor


def file_to_list(filepath: str, sep: str = "\n") -> list[str]:
    return open(filepath, encoding="utf-8").read().strip().split(sep)


class CharDataset(Dataset):
    """A custom PyTorch Dataset for character-level data.

    This dataset processes input strings into 2D tensors and assigns labels
    based on the source of each input line.

    It supports loading data from a directory of `.txt` files, where each file
    represents a class label and its contents are the input samples.

    Parameters
    ----------
    inputs : list[str]
        List of input strings.
    labels : list[str]
        List of string labels corresponding to each input.

    Attributes
    ----------
    input_tensors : list[torch.Tensor]
        List of 2D tensors converted from input strings.
    labels_unique : list[str]
        Unique list of labels.
    labels_indices : list[int]
        Integer index for each label based on its position in `labels_unique`.
    """

    def __init__(
        self,
        inputs: list[str],
        labels: list[str],
        assert_equal_length: bool = True,
    ):
        if assert_equal_length:
            assert len(inputs) == len(
                labels
            ), "`inputs` and `labels have different length."

        super().__init__()
        self.inputs = inputs
        self.labels = labels

        # Map input to tensors.
        self.input_tensors = list(map(line_to_2d_tensor, inputs))

        # Map labels to indices (integers).
        self.labels_unique = list(set(labels))
        self.labels_indices = [self.labels_unique.index(l) for l in labels]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_tensors[index], self.labels_indices[index]

    @classmethod
    def from_dirpath(cls, dirpath: str, ext: str = ".txt") -> CharDataset:
        filepaths = [
            f
            for f in listdir(dirpath)
            if isfile(join(dirpath, f)) and f.endswith(ext)
        ]

        inputs = []
        labels = []

        # For each file, extract its name (the target label) and its content
        # (the input lines).
        for f in filepaths:
            filepath = join(dirpath, f)

            # Input lines.
            new_input = file_to_list(filepath)
            inputs.extend(new_input)
            n_lines = len(new_input)

            # Target data.
            # Current file name is the target label.
            filename = f.removesuffix(".txt")
            labels.extend([filename] * n_lines)

        return CharDataset(inputs, labels)

    @staticmethod
    def collate_fn(batches) -> dict[str, torch.Tensor]:
        """Collate function to combine items into mini-batch for dataloader.

        Parameters
        ----------
        batch : list[tuple[torch.Tensor, torch.Tensor]]:
            list of samples.

        Returns
        -------
        minibatch : tuple[torch.Tensor, torch.Tensor]
        """
        inputs, labels = zip(*batches)
        lengths = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)

        inputs = pad_sequence(inputs, batch_first=True)
        # `inputs` is now a stack of padded tensors of size B x T x N
        # - B: batch size
        # - T: length of the longest sequence
        # - N: features size

        labels = torch.tensor(labels, dtype=torch.long)

        return {"inputs": inputs, "lengths": lengths}, labels
