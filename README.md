# Legal Clause Similarity: Siamese Network Baselines

Two Siamese network models for detecting semantic similarity between legal clauses. Built without pre-trained transformers to serve as baseline architectures for comparison.

## Overview

Legal clauses often express the same meaning using different wording. This project compares two approaches: a bidirectional LSTM model and a CNN with attention mechanism. Both models learn to identify when two clauses are semantically equivalent, which is useful for contract analysis, case law retrieval, and document comparison.

## Architecture

### Siamese BiLSTM
- Bidirectional LSTM encoder with learned word embeddings
- Uses max and average pooling over sequences
- Similarity features computed from concatenated encodings
- 9.99M parameters
- Test accuracy: 99.98%, ROC-AUC: 99.99%

### Siamese CNN with Attention
- Multi-kernel convolutional layers (3, 4, and 5-grams)
- Additive attention mechanism to weight important features
- Extracts n-gram patterns from legal text
- 8.55M parameters
- Test accuracy: 98.73%, ROC-AUC: 99.82%

## Results

| Model | Test Accuracy | Test F1 | ROC-AUC | Training Time |
|-------|--------------|---------|---------|---------------|
| BiLSTM | 99.98% | 99.98% | 99.99% | 41.15 min |
| CNN+Attention | 98.73% | 98.73% | 99.82% | 11.32 min |

Trained on 295,852 clause pairs from 395 legal categories. The dataset includes pairs from the same category (similar) and pairs from different categories (dissimilar).

## Features

- Object-oriented code structure with modular components
- Two baseline architectures for direct comparison
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- GPU training support via CUDA
- Generates negative pairs by mixing clauses from different categories
- Training plots: loss curves, accuracy over epochs, ROC and PR curves
- Confusion matrices for error analysis

## Quick Start

1. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install numpy pandas matplotlib scikit-learn
   ```

2. Prepare your dataset:
   - Create a `dataset/` folder
   - Add CSV files, one per clause category
   - Each CSV should have clause texts and category labels

3. Run the notebook:
   - Open `legal_clause_similarity.ipynb`
   - Run cells in order
   - Results save to the `results/` folder

## Project Structure

```
.
├── legal_clause_similarity.ipynb  # Main notebook
├── dataset/                        # CSV files (one per category)
├── results/                        # Training results and plots
│   ├── bilstm_training.png
│   ├── cnn_training.png
│   ├── bilstm_roc_pr.png
│   ├── cnn_roc_pr.png
│   └── *.csv                      # Metrics and confusion matrices
└── README.md
```

## Technical Details

- Framework: PyTorch
- Word embeddings: Learned from scratch (vocabulary size 40,000, dimension 200)
- Training: Adam optimizer, binary cross-entropy loss, gradient clipping set to 1.0
- Data splits: 70% train, 15% validation, 15% test (stratified)
- Hardware: NVIDIA GPU with CUDA 12.1 or later

## Findings

1. The BiLSTM model performs slightly better, reaching 99.98% accuracy. Bidirectional context helps capture relationships in legal language.

2. The CNN+Attention model trains about 3.6 times faster (11 minutes vs 41 minutes) while still achieving 98.73% accuracy.

3. Using multiple categories (395 in this case) improves training because it provides clear negative examples from different legal domains.

4. Both models work well without pre-trained transformers, showing that baseline architectures can be effective for this task.

## Use Cases

- Comparing legal documents
- Matching contract clauses
- Retrieving similar case law
- Classifying legal clauses
- Finding semantically similar clauses in large legal corpora

## Requirements

- Python 3.8 or later
- PyTorch 2.0 or later (CUDA support recommended for GPU training)
- numpy
- pandas
- matplotlib
- scikit-learn

## License

MIT License

## Contributing

Pull requests are welcome. Open an issue first if you want to discuss major changes.

---

Note: These are baseline models without pre-trained transformers. For transformer-based approaches, you could fine-tune BERT, RoBERTa, or legal domain models like Legal-BERT.

