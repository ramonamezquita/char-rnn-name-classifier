import argparse

import lightning as L
import torch
from dataset import CharDataset
from nn import LigthningWrapper
from torch.utils.data import DataLoader

_CLI_DESCRIPTION = """Predict input string."""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_CLI_DESCRIPTION)
    parser.add_argument(
        "-x", help="Input string to predict.", type=str, required=True
    )
    parser.add_argument("-s", "--sep", help="String separator.", type=str)
    parser.add_argument("-c", "--ckpt", help="Checkpoint path.", type=str)


def predict(model: LigthningWrapper, inputs: list[str]) -> torch.Tensor:

    te_labels = ["unknown"] * len(inputs)  # Dummy labels
    dataset = CharDataset(inputs, te_labels)
    trainer = L.Trainer()
    data_loader = DataLoader(
        dataset, collate_fn=CharDataset.collate_fn, num_workers=10
    )

    # `output_tensors` is a list output tensors. One for each given input.
    output_tensors: list[torch.Tensor] = trainer.predict(
        model, dataloaders=data_loader
    )

    # For each output tensor, return the index of the highest value
    return torch.stack(output_tensors).topk(1)[1].flatten()


def main():

    parser = create_parser()
    args = parser.parse_args()

    # Load model.
    model = LigthningWrapper.load_from_checkpoint(args.ckpt)

    # Create input from string.
    inputs = args.x.split(args.sep)

    # Get unique train labels.
    tr_labels = model.get_params_for("dataset")["labels_unique"]

    # Predict.
    output = predict(model, inputs)
    print(list(map(lambda index: tr_labels[index], output)))
