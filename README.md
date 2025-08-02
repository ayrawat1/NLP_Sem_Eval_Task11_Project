# SemEval 2025 Task 11: Multilingual Emotion Detection


## 🌍 Overview

This repository contains our comprehensive solution for **SemEval 2025 Task 11: "Bridging the Gap in Text-Based Emotion Detection"**. Our approach focuses on multilingual emotion detection across **English, German, and Portuguese Brazilian** using both traditional machine learning baselines and enhanced transformer models.

## ✨ Features

- 🌍 **Multilingual Support**: English, German, Portuguese Brazilian
- 🎯 **Multi-task Learning**: Classification (Track A) + Intensity Prediction (Track B)
- 🤖 **Advanced Models**: Traditional ML baselines + Enhanced BERT/XLM-R transformers
- 📊 **Comprehensive Evaluation**: Cross-lingual performance assessment
- 🔄 **Reproducible Pipeline**: Complete end-to-end workflow
- 📱 **Interactive Notebooks**: Jupyter notebooks for exploration and analysis

## 🚀 Quick Start

### Installation


### Data Preparation

```bash
# Place your SemEval 2025 Task 11 dataset in:
mkdir -p data/raw/dataset
# Copy track_a/, track_b/, track_c/ folders here
```

### Run Complete Pipeline

```bash
# Run the entire pipeline
python scripts/run_full_pipeline.py

# Or run individual steps
python scripts/run_data_exploration.py
python scripts/run_preprocessing.py
python scripts/run_baseline_training.py
python scripts/run_transformer_training.py
python scripts/run_evaluation.py
```


## 🏗️ Architecture

### Pipeline Overview

```
Data Exploration → Preprocessing → Baseline Models → Transformer Models → Evaluation
      ↓                ↓              ↓               ↓                 ↓
   Language         Multilingual    TF-IDF +        BERT +           Comprehensive
   Filtering        Text Cleaning   Traditional ML   Multi-task       Comparison
```

### Model Architecture

- **Baseline Models**: TF-IDF + SVM/Random Forest/Logistic Regression
- **Transformer Models**: BERT-multilingual-cased with enhanced multi-task heads
- **Multi-task Learning**: Simultaneous classification and intensity prediction
- **Cross-lingual Evaluation**: Transfer learning assessment

## 📁 Repository Structure

```
semeval2025-task11-emotion-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── .gitignore                        # Git ignore rules
│
├── data/                             # Data directory
│   ├── raw/                          # Raw SemEval datasets
│   └── processed/                    # Processed datasets
│
├── src/                              # Source code
│   ├── data/                         # Data processing modules
│   │   ├── data_exploration.py       # Data exploration and filtering
│   │   └── preprocessing.py          # Multilingual preprocessing
│   ├── models/                       # Model implementations
│   │   ├── baseline_models.py        # Traditional ML baselines
│   │   └── transformer_models.py     # Enhanced transformer models
│   └── evaluation/                   # Evaluation and metrics
│       └── evaluator.py              # Comprehensive evaluation
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Interactive data exploration
│   ├── 02_preprocessing.ipynb        # Data preprocessing
│   ├── 03_baseline_models.ipynb      # Baseline model training
│   ├── 04_transformer_models.ipynb   # Transformer model training
│   └── 05_evaluation.ipynb           # Model evaluation and comparison
│
├── scripts/                          # Standalone execution scripts
│   ├── run_full_pipeline.py          # Complete pipeline runner
│   ├── run_data_exploration.py       # Data exploration script
│   ├── run_preprocessing.py          # Preprocessing script
│   ├── run_baseline_training.py      # Baseline training script
│   ├── run_transformer_training.py   # Transformer training script
│   └── run_evaluation.py             # Evaluation script
│
├── configs/                          # Configuration files
│   ├── baseline_config.yaml          # Baseline model configuration
│   ├── transformer_config.yaml       # Transformer configuration
│   └── evaluation_config.yaml        # Evaluation configuration
│
├── results/                          # Generated results
│   ├── baseline_results/             # Baseline model outputs
│   ├── transformer_results/          # Transformer model outputs
│   ├── evaluation_results/           # Evaluation reports
│   └── figures/                      # Generated visualizations
│
└── docs/                             # Documentation
    ├── METHODOLOGY.md                # Detailed methodology
    ├── RESULTS.md                    # Complete results analysis
    └── API.md                        # API documentation
```

## 🎯 Methodology

### Data Processing
1. **Language Selection**: English, German, Portuguese Brazilian
2. **Multilingual Preprocessing**: Language-aware text cleaning and normalization
3. **Label Processing**: Binary classification + continuous intensity regression
4. **Cross-lingual Splits**: Training on labeled data, testing on Track C

### Models

#### Baseline Models
- **Features**: TF-IDF with n-grams (1-2), multilingual tokenization
- **Classifiers**: Naive Bayes, SVM, Random Forest, Logistic Regression
- **Multi-output**: MultiOutputClassifier/Regressor for multi-task learning

#### Enhanced Transformer Models
- **Base Model**: BERT-multilingual-cased
- **Architecture**: Dual-head design for classification + regression
- **Training**: Multi-task learning with task-specific loss functions
- **Optimization**: AdamW with linear warmup scheduling

### Evaluation Metrics
- **Classification**: F1-Macro, F1-Micro, per-emotion F1
- **Regression**: Pearson correlation, MSE
- **Cross-lingual**: Transfer performance on Track C

## 🔧 Configuration

### Model Configuration

```yaml
# configs/transformer_config.yaml
model:
  name: "bert-base-multilingual-cased"
  max_length: 128
  dropout_rate: 0.3
  pooling: "cls"

training:
  batch_size: 8
  num_epochs: 3
  learning_rate: 2e-5
  warmup_steps: 100
  weight_decay: 0.01

data:
  languages: ["eng", "deu", "ptbr"]
  emotions: ["joy", "sadness", "fear", "anger", "surprise", "disgust"]
  test_size: 0.2
  random_state: 42
```

### Environment Variables

```bash
# Optional environment variables
export SEMEVAL_DATA_PATH="/path/to/semeval/data"
export CUDA_VISIBLE_DEVICES="0"
export TRANSFORMERS_CACHE="/path/to/cache"
```

##  Performance Analysis

### Key Findings

1. **Multilingual Effectiveness**: BERT-multilingual-cased effectively handles cross-lingual emotion detection
2. **Multi-task Benefits**: Joint training improves both classification and regression performance
3. **Baseline Comparison**: Transformer models show significant improvement over traditional ML approaches
4. **Cross-lingual Transfer**: Strong transfer learning capabilities across language families

### Improvement Strategies

- **Architecture**: Enhanced pooling strategies, deeper task-specific heads
- **Training**: Multi-task loss balancing, learning rate scheduling
- **Data**: Language-aware preprocessing, intensity value normalization

##  Experiments

### Baseline Experiments
```bash
# Run only baseline models
python scripts/run_baseline_training.py --data-path ./data/raw/dataset
```

### Transformer Experiments
```bash
# Run transformer models with custom config
python scripts/run_transformer_training.py --config configs/transformer_config.yaml
```

### Ablation Studies
```bash
# Compare different model configurations
python scripts/run_ablation_study.py --models bert-base,xlm-roberta-base
```

##  Visualization

The evaluation module generates comprehensive visualizations:

- **Performance Comparison**: Baseline vs Transformer models
- **Training Curves**: Loss and metric progression
- **Cross-lingual Analysis**: Transfer learning effectiveness
- **Per-emotion Breakdown**: Detailed emotion-specific performance


##  Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourname2025semeval,
    title={Multilingual Emotion Detection for SemEval 2025 Task 11: Enhanced Transformer Approaches},
    author={Your Name and Collaborators},
    booktitle={Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)},
    year={2025},
    publisher={Association for Computational Linguistics}
}
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SemEval 2025 Task 11 organizers for the multilingual emotion detection challenge
- Hugging Face for the transformers library and pre-trained models
- The multilingual NLP community for inspiration and resources

