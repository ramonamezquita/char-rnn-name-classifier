# Name Classifier — Character-Level RNN (CLI)

A simple and modular **CLI-based PyTorch Lightning project** for classifying names by their origin using a **Character-Level Recurrent Neural Network (RNN)**. The model reads names as sequences of characters and predicts their class (e.g., nationality) using an RNN-based architecture.



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

## Running

Build the image for the project's Docker container environment.

```bash
docker build -t char-rnn-name-classifier .
```

The code is wrapped as a MLflow project, so it  can be run using `mlflow run`.


```bash
docker run char-rnn-name-classifier mlflow run . --env-manager local --entry-point train
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




## Credits

Inspired by the [PyTorch Character-Level RNN tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html), refactored with modern PyTorch + Lightning best practices.
