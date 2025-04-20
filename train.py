import argparse

import lightning as L
from dataset import N_CHARS, CharDataset
from nn import CRITERION, CharRNN, LigthningWrapper
from torch.utils.data import DataLoader, Dataset, random_split

_CLI_DESCRIPTION = """
Training of a basic character-level Recurrent Neural Network (RNN) to classify 
words.
"""


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=_CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dirpath",
        type=str,
        required=True,
        help="[Required] Data directory.",
    )
    parser.add_argument(
        "-r",
        "--rnn",
        type=str,
        help="RNN cell.",
        default="rnn",
        choices=("rnn", "lstm", "gru"),
    )
    parser.add_argument(
        "-s",
        "--hidden_size",
        type=int,
        help="Hidden size.",
        default=16,
    )
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        help="Max number of epochs.",
        default=50,
    )
    parser.add_argument(
        "-v",
        "--val",
        type=float,
        help="Validation split.",
        default=0.2,
    )

    return parser


def train(
    tr: Dataset,
    rnn_cell: str,
    input_size: int,
    output_size: int,
    hidden_size: int = 32,
    batch_size: int = 64,
    max_epochs: int = 50,
    num_workers: int = 10,
    val: None | Dataset = None,
    dataset_params: dict | None = None,
) -> LigthningWrapper:

    if dataset_params is None:
        dataset_params = {}

    dataset_params = {f"dataset__{k}": v for k, v in dataset_params.items()}

    model = LigthningWrapper(
        module=CharRNN,
        criterion=CRITERION,
        module__rnn_cell=rnn_cell,
        module__input_size=input_size,
        module__output_size=output_size,
        module__hidden_size=hidden_size,
        **dataset_params,
    )

    train_loader = DataLoader(
        tr,
        batch_size=batch_size,
        collate_fn=CharDataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = (
        DataLoader(
            val,
            batch_size=batch_size,
            collate_fn=CharDataset.collate_fn,
            num_workers=10,
        )
        if val is not None
        else None
    )

    trainer = L.Trainer(max_epochs=max_epochs)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    return trainer


def main():

    parser = create_parser()
    args = parser.parse_args()
    dataset = CharDataset.from_dirpath(args.dirpath)
    dataset_params = {"labels_unique": dataset.labels_unique}
    output_size = len(dataset.labels_unique)
    input_size = N_CHARS
    tr, val = random_split(dataset, (1 - args.val, args.val))

    train(
        tr,
        rnn_cell=args.rnn,
        input_size=input_size,
        output_size=output_size,
        hidden_size=args.hidden_size,
        max_epochs=args.max_epochs,
        val=val,
        dataset_params=dataset_params,
    )


if __name__ == "__main__":
    main()
