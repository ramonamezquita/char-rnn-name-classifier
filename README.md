# Name Classifier — Character-Level RNN (CLI)

A simple and modular **CLI-based PyTorch Lightning project** for classifying names by their origin using a **Character-Level Recurrent Neural Network (RNN)**. The model reads names as sequences of characters and predicts their class (e.g., nationality) using an RNN-based architecture.

---

## Features

- Simple CLI interface for training and prediction  
- Lightning-powered training loop  
- Custom PyTorch `Dataset` and collation for variable-length sequences  
- Predicts from a checkpointed model with a one-liner CLI call

---


## Dataset Format

Training data should be structured as follows:

```
data/
├── English.txt
├── French.txt
├── Spanish.txt
...
```

Each `.txt` file should:

- Represent a single class (e.g., nationality)
- Contain newline-separated names as examples for that class

Example (`English.txt`):
```
John
Sarah
William
```

---

## 🏋️Training

```bash
python train.py \
  --dirpath ./data \
  --hidden_size 128 \
  --max_epochs 20 \
  --val 0.2
```

---

## Predicting

```bash
python predict.py \
  -x "Satoshi,Nikolai,Ahmed" \
  --sep "," \
  --ckpt path/to/your_checkpoint.ckpt
```


## Architecture

The model structure is:

```
Input (One-hot characters)
      ↓
[RNN / GRU / LSTM]
      ↓
Fully Connected (Linear)
      ↓
LogSoftmax
```

It uses packed padded sequences for efficient processing of variable-length inputs.

---

## Code Structure

```
.
├── dataset.py     # Dataset and preprocessing
├── nn.py          # Model architecture and LightningModule
├── train.py       # CLI training script
├── predict.py     # CLI prediction script
```

---


## Credits

Inspired by the [PyTorch Character-Level RNN tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html), refactored with modern PyTorch + Lightning best practices.
