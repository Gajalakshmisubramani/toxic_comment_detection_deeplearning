# 🛡️ Toxic Comment Detection using Deep Learning

An NLP-based toxic comment detection system developed as my **College Final Year Project** to identify toxic and harmful content in online text.

The project explores and compares multiple machine learning and deep learning approaches for **binary toxic-comment classification** and **multi-label toxicity classification**. It also investigates the use of **K-Max CNN** and a **POS-guided adaptive K-Max CNN** architecture to improve the identification of toxic language.

The models are trained and evaluated primarily using the **Jigsaw Toxic Comment Classification dataset**, with additional cross-dataset evaluation using Twitter hate-speech data.

---

## 🎓 Project Type

**College Final Year Project**

### Domain

* Natural Language Processing (NLP)
* Deep Learning
* Machine Learning
* Text Classification
* Content Moderation
* Artificial Intelligence

---

# 📌 Problem Statement

Online platforms generate a huge amount of user-generated content every day. Some of this content contains toxic, offensive, abusive, or hateful language.

Manually moderating such content is difficult because of:

* The enormous volume of online comments
* Different forms of abusive language
* Context-dependent toxicity
* Multiple categories of harmful content
* Variations in spelling, grammar, and writing style

This project aims to develop automated machine learning and deep learning models capable of identifying toxic comments and classifying them into different toxicity categories.

---

# 🎯 Objectives

The major objectives of the project are:

* Detect whether a comment is toxic or non-toxic.
* Perform multi-label classification of different toxicity categories.
* Compare traditional machine learning with deep learning approaches.
* Investigate CNN-based approaches for toxic text classification.
* Implement K-Max Pooling for retaining important features from text.
* Incorporate Part-of-Speech (POS) information into the K-Max CNN architecture.
* Evaluate model performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* Test model generalization using a separate Twitter hate-speech dataset.

---

# ✨ Key Features

### 🔹 Binary Toxicity Classification

Classifies a comment into:

```text
Non-Toxic
     or
Toxic
```

A binary toxic label is generated when a comment belongs to at least one of the toxicity categories.

### 🔹 Multi-Label Classification

A single comment can belong to multiple categories simultaneously.

The project considers six toxicity labels:

* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

The labels are taken directly from the Jigsaw dataset used in the project.

### 🔹 Multiple Model Comparison

The project evaluates several approaches:

* Logistic Regression
* Bi-LSTM
* CNN
* K-Max CNN
* POS-Guided Adaptive K-Max CNN

### 🔹 Cross-Dataset Evaluation

The trained models are also evaluated on a separate Twitter hate-speech dataset to examine how well the models generalize to unseen data distributions.

---

# 📚 Dataset

## Jigsaw Toxic Comment Classification Dataset

The primary dataset used in this project is the **Jigsaw Toxic Comment Classification** dataset.

Each comment contains six toxicity-related labels:

| Label           | Description                    |
| --------------- | ------------------------------ |
| `toxic`         | General toxic behavior         |
| `severe_toxic`  | Severe toxic behavior          |
| `obscene`       | Obscene language               |
| `threat`        | Threatening language           |
| `insult`        | Insulting language             |
| `identity_hate` | Identity-based hateful content |

The notebook loads the dataset from `train.csv` and creates an additional binary target called `binary_toxic`. A comment is considered toxic when at least one of the six toxicity labels is positive.

---

# 🧹 Data Preparation

The project performs several preprocessing and preparation steps before training.

### Main steps include:

1. Loading the Jigsaw dataset
2. Handling missing comments
3. Creating binary toxicity labels
4. Separating text and target labels
5. Splitting data into training and testing sets
6. Tokenizing text for deep learning models
7. Padding sequences to a fixed length
8. Generating TF-IDF features for Logistic Regression
9. Preparing POS information for the POS-guided models

The main train/test split uses an **80/20 split** with stratification based on the binary toxicity label.

---

# 🤖 Machine Learning Model

## Logistic Regression

Logistic Regression is used as a traditional machine-learning baseline.

The text is converted into numerical features using **TF-IDF**.

The implementation uses:

* TF-IDF vectorization
* Unigrams and bigrams
* English stop-word removal
* Maximum of 100,000 features
* Logistic Regression
* One-vs-Rest classification for multi-label prediction

The notebook also saves the trained Logistic Regression model and TF-IDF vectorizer using `joblib`.

---

# 🧠 Deep Learning Models

## 1. Bi-LSTM

A **Bidirectional Long Short-Term Memory (Bi-LSTM)** network is used to capture contextual relationships in text.

The model architecture includes:

```text
Input Text
     │
     ▼
Embedding Layer
     │
     ▼
Bidirectional LSTM
     │
     ▼
Dropout
     │
     ▼
Dense Layer
     │
     ▼
Prediction
```

The notebook implements both:

* Binary Bi-LSTM
* Multi-label Bi-LSTM

The binary Bi-LSTM evaluation recorded approximately:

| Metric   |     Score |
| -------- | --------: |
| Accuracy |   **96%** |
| ROC-AUC  | **0.958** |

The recorded classification report shows a binary F1-score of approximately **0.77 for the toxic class**.

---

# 2. CNN

A Convolutional Neural Network is used to identify important local patterns and phrases within comments.

The architecture uses:

```text
Input Text
     │
     ▼
Embedding
     │
     ▼
Conv1D
     │
     ▼
Global Max Pooling
     │
     ▼
Dropout
     │
     ▼
Dense
     │
     ▼
Prediction
```

The CNN implementation supports multi-label toxicity classification.

---

# 3. K-Max CNN

The project extends the CNN architecture using **K-Max Pooling**.

Instead of retaining only the single strongest feature, K-Max Pooling retains the top `K` features from the convolution output.

The implementation uses:

```text
Embedding
     │
     ▼
Conv1D
     │
     ▼
K-Max Pooling
     │
     ▼
Dropout
     │
     ▼
Dense
     │
     ▼
Prediction
```

The implementation uses `K = 3` for the K-Max CNN model.

The recorded Binary K-Max CNN evaluation achieved approximately:

| Metric          |    Score |
| --------------- | -------: |
| Accuracy        |  **95%** |
| Toxic Precision | **0.82** |
| Toxic Recall    | **0.71** |
| Toxic F1-score  | **0.76** |

---

# 4. POS-Guided Adaptive K-Max CNN

One of the main experimental components of the project is a **POS-Guided Adaptive K-Max CNN**.

This architecture incorporates **Part-of-Speech (POS)** information into the convolutional representation.

### Architecture

```text
                  Input Text
                     │
                     ▼
                 Embedding
                     │
                     ▼
                   Conv1D
                     │
             ┌───────┴───────┐
             │               │
             ▼               ▼
        Text Features    POS Information
             │               │
             └───────┬───────┘
                     ▼
             POS-Guided Weighting
                     │
                     ▼
          Adaptive K-Max Pooling
                     │
                     ▼
                  Dropout
                     │
                     ▼
                Dense Layer
                     │
                     ▼
                Prediction
```

The implementation contains custom layers for POS-guided weighting and adaptive K-Max pooling. The notebook uses an adaptive K-Max configuration with `k_max=5` and `alpha=1.2`.

This approach is designed to give greater importance to linguistically meaningful parts of the sentence instead of treating every word position equally.

---

# 📊 Model Comparison

The project evaluates multiple approaches across the toxic-comment classification task.

The notebook includes a performance comparison of:

| Model               | Recorded Accuracy |
| ------------------- | ----------------: |
| Logistic Regression |               82% |
| LSTM                |               92% |
| BERT                |               90% |
| K-Max CNN           |               91% |
| K-Max CNN + POS     |               93% |

These values are used in the project's comparison visualization.

> **Note:** The comparison visualization contains manually specified summary values for plotting and should be interpreted as the project's reported comparison rather than as a replacement for the individual model evaluation outputs.

---

# 📈 Evaluation Metrics

The project uses multiple evaluation metrics to assess model performance.

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many predicted toxic comments are actually toxic.

### Recall

Measures how many actual toxic comments are successfully identified.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between different classes across classification thresholds.

For multi-label classification, the project also evaluates **Macro ROC-AUC**.

---

# 🌐 Cross-Dataset Evaluation

To investigate model generalization, the project evaluates trained models on a separate **Twitter hate-speech dataset**.

This is important because a model that performs well on one dataset may not necessarily perform equally well on another dataset with different language patterns and distributions.

The notebook includes cross-dataset evaluation for:

* Logistic Regression
* K-Max CNN
* POS-Adaptive K-Max CNN

The Logistic Regression cross-dataset evaluation records a ROC-AUC of approximately **0.889** on the Twitter hate-speech data.

The project also performs cross-dataset evaluation for the POS-Adaptive K-Max CNN.

---

# 🏗️ Project Workflow

```text
                    Raw Comments
                         │
                         ▼
                  Data Collection
                         │
                         ▼
                  Data Cleaning
                         │
                         ▼
                 Label Preparation
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          TF-IDF                 Tokenization
              │                     │
              ▼                     ▼
      Logistic Regression     Sequence Padding
                                    │
                         ┌──────────┼───────────┐
                         │          │           │
                         ▼          ▼           ▼
                       Bi-LSTM     CNN      K-Max CNN
                                               │
                                               ▼
                                      POS-Adaptive K-Max
                         │
                         ▼
                  Model Evaluation
                         │
                         ▼
          Accuracy / Precision / Recall
                    F1 / ROC-AUC
                         │
                         ▼
              Cross-Dataset Testing
```

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF
* One-vs-Rest Classification

## Deep Learning

* TensorFlow
* Keras
* Bi-LSTM
* CNN
* K-Max CNN
* Custom K-Max Pooling
* Adaptive K-Max Pooling

## Natural Language Processing

* NLTK
* Text Tokenization
* Sequence Padding
* TF-IDF
* POS-based features

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib

## Model Persistence

* Joblib

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Gajalakshmisubramani/toxic_comment_detection_deeplearning.git
```

Navigate into the project:

```bash
cd toxic_comment_detection_deeplearning
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn tensorflow keras nltk matplotlib joblib datasets
```

---

# ▶️ Running the Project

The project is implemented primarily through **Jupyter/Google Colab notebooks**.

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Toxic_comment_deeplearning.ipynb
```

The repository also contains a separate Logistic Regression notebook:

```text
lillogestic_regression.ipynb
```

The main notebook contains approximately **5,500 lines of notebook content** covering preprocessing, model training, evaluation, visualization, and cross-dataset experiments.

---

# 📁 Repository Structure

```text
toxic_comment_detection_deeplearning/
│
├── Toxic_comment_deeplearning.ipynb
│   │
│   ├── Data preprocessing
│   ├── Logistic Regression
│   ├── Bi-LSTM
│   ├── CNN
│   ├── K-Max CNN
│   ├── POS-Guided Adaptive K-Max CNN
│   ├── Model evaluation
│   ├── Cross-dataset evaluation
│   └── Performance visualization
│
├── lillogestic_regression.ipynb
│   │
│   └── Logistic Regression experiments
│
└── README.md
```

The current GitHub repository contains these two notebooks and the README file.

---

# 🔬 Research Component

This project was developed not only as a classification application but also as an experimental study comparing different NLP architectures.

The research focused on investigating whether:

* Sequential models can capture contextual information effectively.
* CNN architectures can identify important local textual patterns.
* K-Max Pooling can retain multiple important features.
* POS information can provide additional linguistic information.
* Adaptive pooling can improve feature selection.
* Models trained on one dataset can generalize to another dataset.

---

# 🎯 Project Outcomes

The project demonstrates that deep learning can effectively be applied to automated toxic-content detection.

The experiments show strong binary classification performance for recurrent and convolutional architectures, while the POS-guided K-Max CNN provides an additional approach for incorporating linguistic information into the model.

The project also demonstrates the importance of **cross-dataset evaluation**, since performance can change when models are exposed to data from a different source.

---

# ⚠️ Limitations

Some limitations of the current project include:

* Toxicity is highly dependent on context.
* Sarcasm and implicit toxicity can be difficult to identify.
* Dataset bias can affect model predictions.
* Minority toxicity categories such as `threat` and `severe_toxic` have fewer examples.
* Performance may decrease when the model is applied to a different platform or domain.
* A high overall accuracy does not necessarily indicate equally strong performance across all toxicity categories.

These limitations highlight the importance of using class-wise metrics and cross-dataset evaluation rather than relying only on accuracy.

---

# 🔮 Future Enhancements

Possible future improvements include:

* 🤖 Fine-tuning modern transformer models such as BERT/RoBERTa
* 🧠 Context-aware toxicity detection
* 🌐 Multilingual toxic-comment detection
* ⚡ Real-time moderation API
* 🔍 Explainable AI for model predictions
* 📊 Interactive model-performance dashboard
* 🧪 More extensive cross-domain evaluation
* ⚖️ Improved handling of class imbalance
* 🚀 Deployment as a web application
* 🔄 Continuous model retraining using new moderation data

---

# 🎓 Learning Outcomes

Through this final-year project, I gained practical experience in:

* Natural Language Processing
* Text preprocessing
* Machine learning classification
* Deep learning
* TensorFlow and Keras
* Recurrent Neural Networks
* Bi-LSTM
* Convolutional Neural Networks
* K-Max Pooling
* POS-based feature integration
* Multi-label classification
* Model evaluation
* Cross-dataset validation
* Python data science workflows
* Research-oriented experimentation

---

# 👩‍💻 Author

**Gajalakshmi Subramani**

**B.E. Computer Science and Engineering (AI & ML)**

**Final Year Project**

---

# ⭐ Repository

Explore the complete implementation:

**Toxic Comment Detection using Deep Learning**

https://github.com/Gajalakshmisubramani/toxic_comment_detection_deeplearning

---

## 📜 Disclaimer

This project was developed for **academic and research purposes** as part of my college final-year project.

The system is intended to demonstrate machine-learning and deep-learning approaches for toxic-content detection and should not be considered a production-ready content-moderation system without further validation, bias analysis, safety testing, and domain-specific evaluation.
