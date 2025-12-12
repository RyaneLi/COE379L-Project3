# Cross-Model Comparison for News Topic Classification: Classical Machine Learning vs. Fine-Tuned Transformers

**COE379L Project 3**  
**Advanced Classical Algorithms vs. Fine-Tuned Transformers**

---

## 1. Introduction and Project Statement

### 1.1 Background

Text classification remains one of the fundamental tasks in natural language processing (NLP), with applications spanning from content moderation to automated news categorization. The field has witnessed a paradigm shift from classical machine learning approaches, which rely on hand-engineered features like TF-IDF, to modern transformer-based models that leverage pre-trained contextual embeddings. This evolution raises critical questions about the trade-offs between computational efficiency and predictive performance.

### 1.2 Project Objective

This project conducts a comprehensive, direct comparison between two distinct methodological paradigms for multi-class text classification:

1. **Classical Feature-Engineered Models**: Utilizing traditional machine learning algorithms (XGBoost and Support Vector Machines) with TF-IDF feature extraction
2. **Fine-Tuned Transformer Models**: Employing state-of-the-art pre-trained language models (RoBERTa-base) fine-tuned on the target task

The primary goal is to determine the optimal modeling approach by systematically evaluating both computational efficiency and predictive performance on a standardized news classification task.

### 1.3 Research Questions

This study addresses three key research questions:

1. **Performance Gap Analysis**: How significant is the performance difference between classical, feature-engineered models and fine-tuned transformer models on a modern news classification task?

2. **Computational Trade-offs**: What are the trade-offs in training time and inference latency for each methodological approach?

3. **Practical Recommendations**: Can a resource-efficient classical model provide sufficient performance to justify avoiding the higher computational costs associated with fine-tuning large transformer models?

### 1.4 Scope and Limitations

This study focuses on the AG News dataset, a balanced four-class news classification task. The comparison is limited to:
- Classical models: XGBoost and SVM (Linear and RBF kernel)
- Transformer model: RoBERTa-base
- Evaluation metrics: Accuracy, Macro-Averaged F1-Score, Log Loss, Training Time, and Inference Latency

The findings are specific to this dataset and task, though the methodology can be generalized to similar text classification problems.

---

## 2. Data Sources and Technologies Used

### 2.1 Dataset: AG News

The AG News dataset, sourced from the Hugging Face Datasets Hub, serves as the standardized benchmark for this comparison. This dataset was selected for its:

- **Scale**: Over 120,000 training samples and 7,600 test samples
- **Balance**: Four balanced categories (World, Sports, Business, Sci/Tech)
- **Pre-split Structure**: Pre-divided into training and test sets, ensuring reproducibility
- **Format**: Each sample contains a news article title and description, which are combined into a single text input for classification consistency

The dataset's balanced nature makes macro-averaged metrics appropriate for evaluation, as they treat all classes equally regardless of their individual sizes.

### 2.2 Technology Stack

#### 2.2.1 Classical Models Pipeline

**Libraries and Frameworks:**
- **scikit-learn** (v1.3+): Primary framework for classical machine learning algorithms
  - `TfidfVectorizer`: TF-IDF feature extraction
  - `XGBClassifier`: XGBoost gradient boosting implementation
  - `LinearSVC` and `SVC`: Support Vector Machine implementations
  - `RandomizedSearchCV`: Hyperparameter optimization with cross-validation
- **XGBoost** (v3.1+): Standalone gradient boosting library integrated with scikit-learn
- **pandas** and **numpy**: Data manipulation and numerical computations
- **joblib**: Model persistence and serialization

**Hardware Considerations:**
- Classical models were trained on CPU
- Sparse matrix representations used for memory efficiency with TF-IDF features
- Hyperparameter search optimized for computational efficiency

#### 2.2.2 Transformer Models Pipeline

**Libraries and Frameworks:**
- **PyTorch**: Deep learning framework for model training and inference
- **Hugging Face Transformers**: Pre-trained model library and training utilities
  - `AutoTokenizer` and `AutoModelForSequenceClassification`: Model and tokenizer loading
  - `Trainer` and `TrainingArguments`: Streamlined training loop management
  - `EarlyStoppingCallback`: Training optimization
- **Hugging Face Datasets**: Efficient dataset loading and preprocessing
- **scikit-learn**: Evaluation metrics (accuracy, F1-score, log loss)

**Hardware Considerations:**
- RoBERTa-base fine-tuning supports GPU acceleration (CUDA)
- Mixed precision training (FP16) enabled when GPU available
- Model size: ~125 million parameters

#### 2.2.3 Development Environment

- **Jupyter Notebooks**: Interactive development and analysis
- **Python 3.13**: Programming language
- **Git**: Version control and collaboration
- **Virtual Environment**: Dependency isolation using `venv`

### 2.3 Data Preprocessing

The AG News dataset's `text` field already contains combined title and description information. The preprocessing pipeline:

1. **Text Preparation**: Directly uses the `text` field as the input feature
2. **Tokenization (Classical)**: TF-IDF vectorization with unigrams and bigrams
3. **Tokenization (Transformer)**: RoBERTa tokenizer with max length of 512 tokens, truncation, and padding

No additional text cleaning (e.g., lowercasing, punctuation removal) was performed to preserve the original text characteristics and allow each model type to handle preprocessing according to its design.

---

## 3. Methods Employed

### 3.1 Classical Models Pipeline

#### 3.1.1 Feature Extraction: TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) was selected as the feature extraction method for classical models due to its effectiveness in capturing term importance within documents relative to the corpus.

**Configuration:**
- **N-grams**: Unigrams (1-grams) and bigrams (2-grams) to capture both word-level and phrase-level patterns
- **Vocabulary Size**: Limited to top 50,000 features to balance representational power and computational efficiency
- **Document Frequency**: Minimum document frequency (`min_df=2`) to filter rare terms, maximum document frequency (`max_df=0.95`) to exclude overly common terms
- **Stop Words**: English stop words removed to focus on content-bearing terms

The TF-IDF vectorization produces high-dimensional sparse matrices, which are efficiently handled by scikit-learn's sparse matrix implementations.

#### 3.1.2 Model 1: XGBoost

XGBoost (eXtreme Gradient Boosting) was selected as a representative of ensemble tree-based methods, known for their strong performance on structured and text classification tasks.

**Model Architecture:**
- Gradient boosting decision trees
- Integrated with scikit-learn API for consistency

**Hyperparameter Optimization:**
- **Search Method**: RandomizedSearchCV with 2-fold cross-validation
- **Search Space**:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [5, 7, 9]
  - `learning_rate`: [0.1, 0.2, 0.3]
  - `subsample`: [0.8, 0.9, 1.0]
- **Optimization Metric**: Macro-averaged F1-Score
- **Search Efficiency**: Conducted on a 5,000-sample subset to reduce computational time, with final model trained on full dataset

**Final Model Training:**
- Trained on full training set with optimized hyperparameters
- Sparse TF-IDF matrices converted to dense format for XGBoost compatibility

#### 3.1.3 Model 2: Support Vector Machines

Two SVM variants were implemented to explore different kernel strategies:

**A. LinearSVC (Linear Support Vector Classifier)**
- **Kernel**: Linear (implicit)
- **Advantages**: Fast training and inference, memory efficient
- **Hyperparameter Optimization**: RandomizedSearchCV
  - `C`: [0.1, 1.0, 10.0, 100.0]
  - `max_iter`: [1000, 2000]
- **Training**: Full dataset on sparse TF-IDF features

**B. SVC with RBF Kernel**
- **Kernel**: Radial Basis Function (non-linear)
- **Advantages**: Can capture non-linear decision boundaries
- **Disadvantages**: Computationally expensive, requires dense feature matrices
- **Hyperparameter Optimization**: RandomizedSearchCV on 2,000-sample subset
  - `C`: [0.1, 1.0, 10.0]
  - `gamma`: ['scale', 'auto', 0.001, 0.01]
- **Training**: Trained on 2,000-sample subset due to computational constraints

#### 3.1.4 Optimization Strategy

All classical models employed RandomizedSearchCV for hyperparameter tuning:
- **Cross-Validation**: 2-fold CV to balance robustness and speed
- **Scoring Metric**: Macro-averaged F1-Score
- **Random State**: Fixed (42) for reproducibility
- **Caching**: Best hyperparameters cached to skip repeated searches in subsequent runs

### 3.2 Transformer Models Pipeline

#### 3.2.1 Base Model: RoBERTa-base

RoBERTa (Robustly Optimized BERT Pretraining Approach) was selected over BERT for several reasons:

- **Improved Pre-training**: Dynamic masking strategy and larger batch sizes
- **No Next Sentence Prediction**: Simplified architecture focused on masked language modeling
- **Better Downstream Performance**: Typically yields superior results on classification tasks
- **Model Size**: 125 million parameters, providing a good balance between capacity and efficiency

#### 3.2.2 Fine-Tuning Methodology

**Training Configuration:**
- **Epochs**: 3 epochs with early stopping (patience=2)
- **Batch Size**: 16 samples per device (train), 32 samples per device (eval)
- **Learning Rate Schedule**: Linear warmup over 500 steps, followed by decay
- **Weight Decay**: 0.01 for regularization
- **Mixed Precision**: FP16 enabled when GPU available
- **Evaluation Strategy**: Evaluate every 500 steps during training

**Tokenization:**
- **Tokenizer**: RoBERTa tokenizer (Byte-Pair Encoding)
- **Max Length**: 512 tokens (RoBERTa's maximum)
- **Truncation**: Long sequences truncated to max length
- **Padding**: Sequences padded to max length for batch processing

**Training Process:**
1. Load pre-trained RoBERTa-base from Hugging Face
2. Add classification head (dense layer mapping to 4 classes)
3. Fine-tune entire network end-to-end on AG News training set
4. Use test set for evaluation during training (monitoring overfitting)
5. Save best model based on macro-averaged F1-Score

#### 3.2.3 Contextual Embeddings

Unlike TF-IDF's static word representations, RoBERTa generates contextual embeddings:
- **Dynamic Representations**: Word meanings adapt based on surrounding context
- **Subword Tokenization**: Handles out-of-vocabulary words through subword units
- **Pre-trained Knowledge**: Leverages knowledge from pre-training on large text corpora

### 3.3 Evaluation Methodology

#### 3.3.1 Performance Metrics

All models were evaluated using consistent metrics on the same test set:

1. **Accuracy**: Proportion of correctly classified samples
2. **Macro-Averaged F1-Score**: Unweighted mean of per-class F1-scores (appropriate for balanced dataset)
3. **Log Loss**: Logarithmic loss for probabilistic predictions (measures prediction confidence)

#### 3.3.2 Efficiency Metrics

Computational efficiency was measured through:

1. **Training Time**: Total wall-clock time for model training (including hyperparameter search where applicable)
2. **Inference Latency**: Time to process 1,000 test samples (measured after model warm-up)

#### 3.3.3 Experimental Design

- **Reproducibility**: Fixed random seeds (42) across all experiments
- **Fair Comparison**: All models evaluated on identical test set
- **Hardware Consistency**: Classical models on CPU, transformer on GPU (when available)
- **Progress Tracking**: Comprehensive logging implemented for all long-running operations

---

## 4. Results

### 4.1 Performance Metrics

[**Note**: Actual numerical results will be inserted here after running the notebooks. The following structure provides placeholders for the results table and analysis.]

#### 4.1.1 Quantitative Results Table

| Model | Accuracy | Macro F1-Score | Log Loss | Training Time (s) | Inference Latency per 1k (s) |
|-------|----------|----------------|----------|-------------------|------------------------------|
| XGBoost | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SVM-LinearSVC | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SVM-RBF | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| RoBERTa-base | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

*Table 1: Comprehensive performance and efficiency metrics for all models*

#### 4.1.2 Performance Analysis

**Accuracy Comparison:**
[Analysis of accuracy results will be provided here, comparing classical vs. transformer models]

**F1-Score Analysis:**
[Analysis of macro-averaged F1-scores, discussing class-wise performance and overall model effectiveness]

**Log Loss Analysis:**
[Discussion of prediction confidence and calibration, noting that LinearSVC does not provide probability estimates]

### 4.2 Efficiency Metrics

#### 4.2.1 Training Time Analysis

[Analysis of training time differences between models, including:
- Classical models: Fast training, especially LinearSVC
- Transformer: Longer training time due to model complexity and fine-tuning process
- Hyperparameter search overhead for classical models]

#### 4.2.2 Inference Latency Analysis

[Analysis of inference speed:
- Classical models: Very fast inference, suitable for real-time applications
- Transformer: Slower inference due to model size and complexity
- Practical implications for deployment scenarios]

### 4.3 Performance vs. Efficiency Trade-offs

[Comprehensive discussion of the trade-offs:
- When to choose classical models: Resource-constrained environments, real-time applications, interpretability needs
- When to choose transformers: Maximum accuracy requirements, sufficient computational resources, complex language understanding needs
- Cost-benefit analysis]

### 4.4 Visualizations

[Placeholders for figures that will be added later:]

**Figure 1**: Confusion Matrix - Best Classical Model (XGBoost or SVM-LinearSVC)  
*[To be inserted: Shows per-class classification performance]*

**Figure 2**: Confusion Matrix - RoBERTa-base  
*[To be inserted: Shows per-class classification performance for transformer model]*

**Figure 3**: Macro F1-Score Comparison - All Models  
*[To be inserted: Bar chart comparing F1-scores across all models]*

**Figure 4**: Training Time Comparison - All Models  
*[To be inserted: Bar chart (log scale) comparing training times]*

**Figure 5**: Performance vs. Efficiency Scatter Plot  
*[To be inserted: F1-Score vs. Training Time visualization]*

### 4.5 Key Findings

[Summary of key findings will be provided here, including:
1. Performance gap magnitude between classical and transformer models
2. Computational cost differences
3. Practical recommendations based on use case]

---

## 5. Discussion and Conclusion

### 5.1 Research Questions Revisited

#### 5.1.1 Performance Gap Analysis

[Discussion of how significant the performance difference is between classical and transformer models, with specific numerical comparisons and statistical significance considerations]

#### 5.1.2 Computational Trade-offs

[Detailed analysis of training time and inference latency trade-offs, including:
- Absolute time differences
- Relative efficiency ratios
- Scalability considerations]

#### 5.1.3 Practical Recommendations

[Recommendations for different scenarios:
- **High-accuracy requirements with sufficient resources**: RoBERTa-base
- **Real-time applications with moderate accuracy needs**: XGBoost or LinearSVC
- **Resource-constrained environments**: LinearSVC
- **Balanced approach**: XGBoost (good accuracy with reasonable efficiency)]

### 5.2 Limitations and Future Work

**Limitations:**
- Single dataset evaluation (AG News)
- Limited to four-class classification
- Hardware-specific results (CPU vs. GPU)
- Hyperparameter search space could be expanded

**Future Work:**
- Evaluation on additional datasets and tasks
- Comparison with other transformer architectures (BERT, DistilBERT)
- Ensemble methods combining classical and transformer models
- Quantization and model compression for transformer efficiency
- Analysis of computational cost in cloud deployment scenarios

### 5.3 Conclusion

[Final conclusion summarizing:
- The performance-efficiency trade-offs observed
- The practical implications for real-world deployment
- The value of both classical and transformer approaches in different contexts
- Overall project contributions to understanding model selection for text classification]

---

## 6. References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

2. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273-297.

3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171-4186.

4. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*.

5. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

6. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.

7. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38-45.

8. Hugging Face. (2024). *AG News Dataset*. Retrieved from https://huggingface.co/datasets/ag_news

9. Hugging Face. (2024). *Transformers Library Documentation*. Retrieved from https://huggingface.co/docs/transformers

---

## Appendix A: Hyperparameter Configurations

### A.1 XGBoost Best Hyperparameters
- `n_estimators`: [TBD]
- `max_depth`: [TBD]
- `learning_rate`: [TBD]
- `subsample`: [TBD]

### A.2 SVM-LinearSVC Best Hyperparameters
- `C`: [TBD]
- `max_iter`: [TBD]

### A.3 SVM-RBF Best Hyperparameters
- `C`: [TBD]
- `gamma`: [TBD]

### A.4 RoBERTa Training Configuration
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `learning_rate`: [default from TrainingArguments]
- `warmup_steps`: 500
- `weight_decay`: 0.01

---

**Report Generated**: [Date]  
**Project Repository**: https://github.com/RyaneLi/COE379L-Project3

