from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

CRITERION = torch.nn.NLLLoss


def rnn_factory(
    name: str, input_size: int, hidden_size: int, **kwargs
) -> nn.Module:
    """Returns RNN cell."""

    name_to_rnn = {
        "rnn": nn.RNN,
        "gru": nn.GRU,
        "lstm": nn.LSTM,
    }

    try:
        cls = name_to_rnn[name]
    except KeyError:
        raise ValueError(
            f"Invalid RNN `{name}`. Available options are: {tuple(name_to_rnn)}."
        )

    return cls(input_size=input_size, hidden_size=hidden_size, **kwargs)


class LigthningWrapper(L.LightningModule):

    def __init__(
        self,
        module: type[nn.Module],
        criterion: type[nn.Module],
        optimizer: type[torch.optim.Optimizer] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.kwargs = kwargs

    def configure_model(self) -> None:
        # Step 1: Train model and save checkpoint
        # ----------------------------------------
        # model = LightningWrapper(...)
        # trainer.fit(model)
        # Lightning saves:
        # - model.state_dict()  → includes self.module_.*
        # - model hyperparameters (via self.save_hyperparameters())

        # Step 2: Load model from checkpoint
        # ----------------------------------

        # 2.1: __init__ is called with saved hyperparameters
        # - self.module, self.criterion, etc. are set up (but self.module_ is not yet instantiated)

        # 2.2: configure_model() is called automatically when loading from checkpoint.
        # - self.module_ = self.module(**params) → new instance of the inner nn.Module
        # - self.criterion_ = self.criterion(**params)

        # 2.3: Lightning loads the checkpoint's state_dict
        # - self.load_state_dict(checkpoint["state_dict"]) is called internally
        # - weights are matched by parameter names (e.g., "module_.linear.weight")
        # - self.module_ now contains the trained weights from the checkpoint

        if not hasattr(self, "module_"):
            self.module_ = self.module(**self.get_params_for("module"))
        if not hasattr(self, "criterion_"):
            self.criterion_ = self.criterion(**self.get_params_for("criterion"))

    def get_params_for(self, prefix: str) -> dict:
        if not prefix.endswith("__"):
            prefix += "__"

        return {
            key.removeprefix(prefix): val
            for key, val in self.kwargs.items()
            if key.startswith(prefix)
        }

    def forward(self, x: Any) -> torch.Tensor:
        return self.module_(x)

    def training_step(self, batch, batch_index) -> torch.Tensor:
        x, y = batch
        loss = self.criterion_(self(x), y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_index) -> None:
        x, y = batch
        loss = self.criterion_(self(x), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        x, y = batch
        return self(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return (
            self.optimizer
            if self.optimizer is not None
            else torch.optim.Adam(self.parameters(), lr=1e-3)
        )


class CharRNN(nn.Module):
    """CharRNN

    This CharRNN class implements an RNN with three components:
                    RNN -> Linear -> LogSoftmax

    Parameters
    ----------

    """

    def __init__(
        self,
        rnn_cell: str,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        super().__init__()
        self.rnn = rnn_factory(
            rnn_cell, input_size=input_size, hidden_size=hidden_size
        )
        self.h2o = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_dict: dict[str, torch.Tensor]
            Dictionary with keys:
                - inputs: torch.Tensor (shape B x T x F)
                    Contains one hot encoded zero-padded character lines.
                    Each line is represented by a T x F matrix where T
                    is the number of characters and F the dimension of the
                    embedding space.

                - lengths: torch.Tensor (shape B)
                    Lenght of the original lines (before padding).
                    Used for packing padded sequences.

            where:
                B: batch size.
                T: sequence length (number of characters in each line).
                F: feature size (vocabulary size / number of allowed characters).

        Returns
        -------
        logprobs: torch.Tensor
            Log-probabilities of each class.
        """

        packed_input = pack_padded_sequence(
            x_dict["inputs"],
            x_dict["lengths"],
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed_input)
        output = self.h2o(hidden[0])
        logprobs = self.softmax(output)
        return logprobs
