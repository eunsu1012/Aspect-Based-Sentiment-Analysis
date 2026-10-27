# 💬 Aspect-Based Sentiment Analysis (ABSA) on Healthcare Drug Reviews

An NLP project for fine-grained sentiment analysis of patient drug reviews, developed as my Master's thesis in Data Science.  
This project classifies sentiment **separately for three aspects** of each review and compares 5 models across machine learning, deep learning, and transformer paradigms.

## 📌 Overview

Traditional sentiment analysis assigns a single sentiment (positive, negative, neutral) to an entire text.  
However, patient reviews often contain different opinions about different aspects of the same drug.

For example:  
"Zoloft helped me feel calmer and more in control, but I had headaches for the first week."

- Benefits → Positive  
- Side Effects → Negative  

This project classifies sentiment for each aspect (Comments, Benefits, Side Effects) individually, instead of assigning one label to the whole review.


## 🎯 Objectives

- Perform aspect-level sentiment classification on healthcare drug reviews  
- Compare traditional ML, deep learning, and transformer models under a consistent evaluation setup  
- Handle class imbalance and evaluate models fairly with Macro-F1  
- Identify which model types work best for each aspect, and why  


## 🧠 Key Features

- Aspect-level sentiment classification for 3 aspects: Comments, Benefits, Side Effects  
- 5 models × 3 aspects = **15 experiments**  
- DistilBERT sentence-pair input (`[CLS] review [SEP] aspect [SEP]`) to condition predictions on each aspect  
- Stratified nested cross-validation with **1,800 model fits** for classical models  


## 🗂️ Dataset

**UCI Drug Review Dataset (Druglib.com)**, loaded via `ucimlrepo` (`id=461`)

- ~4,100 patient reviews → **12,000+ aspect-level samples**  
- Each review has separate text fields for each aspect, with labels mapped as follows:

| Aspect | Text column | Label source | Negative | Neutral | Positive |
|---|---|---|---|---|---|
| Comments | `commentsReview` | `rating` (1–10) | 1–3 | 4–7 | 8–10 |
| Benefits | `benefitsReview` | `effectiveness` | Ineffective | Moderately / Marginally Effective | Highly / Considerably Effective |
| Side Effects | `sideEffectsReview` | `sideEffects` | Moderate / Severe / Extremely Severe | Mild | No Side Effects |


## ⚙️ Methodology

1. Text preprocessing  
   - Unicode normalization and lowercasing  
   - Replacing URLs, user handles, and numbers with `<url>`, `<user>`, `<num>`  
   - Keeping `!`, `?`, `'` for negation and sentiment cues  

2. Feature Representation  
   - Classical ML: TF-IDF (word 1–3-grams, max 30K features)  
   - Deep learning: trainable word embeddings  
   - Transformer: sentence-pair tokenization (review + aspect name)  

3. Sentiment Classification  
   - **Classical ML**: Linear SVM, RBF-SVC, XGBoost (scikit-learn)  
   - **Deep learning**: CNN-BiLSTM (TensorFlow/Keras)  
   - **Transformer**: DistilBERT (`distilbert-base-uncased`, Hugging Face)  
   - Class imbalance handled with `class_weight="balanced"`  

4. Validation  
   - Classical ML: stratified nested CV (5 outer × 3 inner folds), `GridSearchCV` over 39 hyperparameter configurations  
   - CNN-BiLSTM: stratified 5-fold CV  
   - DistilBERT: single stratified 80/20 split (due to compute limits)  


## 📊 Model Evaluation

The models are evaluated using:

- **Macro-F1** (primary metric): averages F1 across classes equally, so minority classes count as much as the majority class  
- **Accuracy**  
- **Precision / Recall** per class  
- **Confusion matrices** to analyze class-wise errors  


## 🛠️ Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn / XGBoost  
- TensorFlow / Keras  
- Hugging Face Transformers  
- Jupyter Notebook  


## 📁 Project Structure

<!-- 실제 저장소 폴더 구조에 맞게 수정하세요 -->

```
Aspect-Based-Sentiment-Analysis/
│
├── data/                # Dataset
├── notebooks/           # EDA & modeling
├── src/                 # Preprocessing & nested CV pipeline
├── results/             # Metrics, confusion matrices, figures
└── README.md
```


## 🚀 Results

**Macro-F1 / Accuracy by aspect**

| Model | Comments | Benefits | Side Effects | Mean Macro-F1 |
|---|---|---|---|---|
| Linear SVM | 0.500 / 0.568 | **0.625** / 0.751 | 0.715 / 0.715 | 0.613 |
| RBF-SVC | 0.500 / 0.574 | 0.624 / 0.753 | 0.744 / 0.741 | **0.623** |
| XGBoost | 0.431 / 0.572 | 0.563 / 0.754 | 0.728 / 0.726 | 0.574 |
| CNN-BiLSTM | 0.420 / 0.512 | 0.538 / 0.707 | 0.701 / 0.701 | 0.553 |
| DistilBERT | **0.503** / 0.574 | 0.615 / 0.760 | **0.752** / 0.747 | **0.623** |

- **DistilBERT** achieved the best Side Effects Macro-F1 (**75.2%**, +5.1 pts over CNN-BiLSTM)  
- **RBF-SVC** matched DistilBERT's mean Macro-F1 (**62.3%**), and **Linear SVM** led on Benefits, showing that tuned TF-IDF baselines remain competitive with a 66M-parameter transformer on small domain datasets  
- **Neutral** sentiment was the hardest class for every model  

> Note: DistilBERT was evaluated on a single split, while the other models report means over 5 folds, so the comparison is not strictly equal.


## 📌 Limitations

- Comments labels come from overall ratings, which don't always match the comment text (label noise)  
- Mixed or contrastive phrasing ("worked, but…") causes most neutral-class errors  
- Results are based on a single, relatively small dataset; cross-domain performance was not tested  


## 🔧 Future Work

- Cross-validate DistilBERT for a fully comparable evaluation  
- Test larger and domain-specific transformers (e.g., clinical BERT variants)  
- Extend to cross-domain and multilingual ABSA  
- Build a real-time ABSA system for review platforms  


## 👤 Author

Park, Eunsoo


## 🎓 Academic Context

This project was developed as my Master's thesis in Data Science at Minnesota State University, Mankato (2025), advised by Dr. Rushit Dave.

**Thesis:** *Evaluating Aspect-Based Sentiment Analysis in Healthcare Drug Reviews Across Machine Learning, Deep Neural Networks, and Transformer Models*


## ✔️ Summary

> An aspect-level sentiment classification study on healthcare drug reviews, showing that a transformer (DistilBERT) leads on Side Effects while tuned classical models (SVC) match its overall performance on a small domain dataset
