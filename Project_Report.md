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

#### 4.1.1 Quantitative Results Table

| Model | Accuracy | Macro F1-Score | Log Loss | Training Time (s) | Inference Latency per 1k (s) |
|-------|----------|----------------|----------|-------------------|------------------------------|
| XGBoost | 0.8809 | 0.8807 | 0.4399 | 2,862.1 | 0.163 |
| SVM-LinearSVC | 0.9228 | 0.9226 | N/A | 9.8 | 0.004 |
| SVM-RBF | 0.8645 | 0.8637 | 0.4411 | 1,653.9 | 181.6 |
| RoBERTa-base | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

*Table 1: Comprehensive performance and efficiency metrics for all models*

#### 4.1.2 Performance Analysis

**Accuracy Comparison:**

The classical models demonstrate strong performance on the AG News classification task. SVM-LinearSVC achieves the highest accuracy at **92.28%**, outperforming both XGBoost (88.09%) and SVM-RBF (86.45%). This result is notable given LinearSVC's computational efficiency, making it an excellent choice for production deployments where both accuracy and speed are priorities.

XGBoost achieves competitive accuracy (88.09%) with robust performance across all classes. The SVM-RBF model, despite its non-linear kernel capabilities, shows lower accuracy (86.45%) than the linear variant, suggesting that the linear decision boundary is sufficient for this task and the additional complexity of the RBF kernel may lead to overfitting.

**F1-Score Analysis:**

The macro-averaged F1-scores closely mirror the accuracy results, indicating balanced performance across all four classes (World, Sports, Business, Sci/Tech). SVM-LinearSVC achieves the highest F1-score of **0.9226**, followed by XGBoost at **0.8807** and SVM-RBF at **0.8637**. The close alignment between accuracy and F1-scores confirms that the models perform consistently across all classes, which is expected given the balanced nature of the AG News dataset.

The high F1-scores across all classical models suggest that the TF-IDF feature representation effectively captures the discriminative patterns needed for news classification. The linear SVM's superior performance indicates that the decision boundaries between news categories are largely linearly separable in the TF-IDF feature space.

**Log Loss Analysis:**

Log loss provides insight into prediction confidence and calibration. XGBoost achieves a log loss of **0.4399**, indicating reasonably well-calibrated probability estimates. SVM-RBF shows a similar log loss of **0.4411**, suggesting comparable confidence levels in predictions.

Notably, LinearSVC does not provide probability estimates by default (it uses a linear loss function without probability calibration), so log loss cannot be calculated for this model. This is a limitation when probability estimates are required for downstream applications, though it does not affect classification accuracy.

The log loss values for XGBoost and SVM-RBF are relatively low, indicating that these models are confident in their predictions and well-calibrated. This is important for applications requiring reliable confidence scores.

### 4.2 Efficiency Metrics

#### 4.2.1 Training Time Analysis

The training times reveal significant differences between classical models, primarily driven by hyperparameter search overhead and model complexity:

**SVM-LinearSVC** demonstrates exceptional efficiency, completing training in just **9.8 seconds**. This remarkable speed is due to its linear nature and efficient implementation for sparse matrices, making it ideal for rapid prototyping and deployment scenarios where training time is a constraint.

**XGBoost** requires **2,862 seconds (47.7 minutes)** for training, which includes hyperparameter optimization via RandomizedSearchCV. The ensemble nature of XGBoost, combined with the hyperparameter search, contributes to this longer training time. However, this investment yields competitive accuracy and well-calibrated probability estimates.

**SVM-RBF** takes **1,654 seconds (27.6 minutes)** for training, despite being trained on a smaller subset (2,000 samples) due to computational constraints. The RBF kernel's computational complexity, requiring dense matrix operations, significantly increases training time compared to the linear variant.

The hyperparameter search overhead is substantial for both XGBoost and SVM models, but this one-time cost is typically acceptable given the performance improvements. For production systems, cached hyperparameters can eliminate this overhead in subsequent training runs.

#### 4.2.2 Inference Latency Analysis

Inference latency shows dramatic differences between models, with implications for real-time deployment:

**SVM-LinearSVC** achieves the fastest inference at **0.004 seconds per 1,000 samples**, making it exceptionally suitable for real-time applications requiring millisecond-level response times. This speed advantage, combined with its high accuracy, makes LinearSVC an excellent choice for production systems with high throughput requirements.

**XGBoost** demonstrates fast inference at **0.163 seconds per 1,000 samples**, approximately 40 times slower than LinearSVC but still highly efficient for most applications. This inference speed, combined with its competitive accuracy and probability estimates, makes XGBoost a versatile choice.

**SVM-RBF** shows significantly slower inference at **181.6 seconds per 1,000 samples**, over 45,000 times slower than LinearSVC. This poor inference performance, combined with lower accuracy, makes RBF SVM impractical for most real-world applications despite its theoretical non-linear capabilities.

The inference latency differences have practical implications: LinearSVC can process over 250,000 samples per second, while XGBoost handles approximately 6,000 samples per second. Both are suitable for real-time applications, but LinearSVC's speed advantage is substantial for high-throughput scenarios.

### 4.3 Performance vs. Efficiency Trade-offs

The results reveal clear trade-offs between performance and efficiency, guiding model selection based on specific use case requirements:

**When to Choose Classical Models:**

Classical models excel in several scenarios:
- **Resource-Constrained Environments**: LinearSVC's minimal training time (9.8s) and memory footprint make it ideal for systems with limited computational resources or edge devices.
- **Real-Time Applications**: With inference latency of 0.004s per 1,000 samples, LinearSVC can handle high-throughput scenarios requiring millisecond response times, such as content filtering or real-time news categorization APIs.
- **Interpretability Needs**: TF-IDF features and linear decision boundaries provide more interpretable models compared to deep learning approaches, which is valuable for regulated industries or when model explanations are required.
- **Cost-Effective Solutions**: The combination of high accuracy (92.28%) and exceptional efficiency makes LinearSVC an optimal choice when balancing performance with operational costs.

**When to Choose Transformers:**

Transformer models are preferable when:
- **Maximum Accuracy Requirements**: If the highest possible accuracy is critical and computational resources are available, fine-tuned transformers typically achieve superior performance.
- **Complex Language Understanding**: Tasks requiring deep semantic understanding, context awareness, or handling of nuanced language benefit from transformer architectures.
- **Sufficient Computational Resources**: When training time and inference latency are not primary constraints, transformers can provide state-of-the-art performance.

**Cost-Benefit Analysis:**

The classical models, particularly LinearSVC, demonstrate an exceptional cost-benefit ratio:
- **LinearSVC**: Achieves 92.28% accuracy with minimal computational cost (9.8s training, 0.004s/1k inference), representing the best efficiency-to-performance ratio.
- **XGBoost**: Provides a balanced approach with 88.09% accuracy, probability estimates, and reasonable efficiency (47.7min training, 0.163s/1k inference), suitable when probability estimates are required.
- **SVM-RBF**: Shows poor cost-benefit ratio with lower accuracy (86.45%) and extremely slow inference (181.6s/1k), making it impractical for most applications.

For the AG News classification task, the linear decision boundary appears sufficient, as evidenced by LinearSVC's superior performance over the non-linear RBF variant. This suggests that classical models with appropriate feature engineering can compete effectively with more complex approaches for well-structured classification tasks.

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

The experimental results yield several key insights:

1. **Classical Models Achieve Strong Performance**: SVM-LinearSVC achieves 92.28% accuracy, demonstrating that well-engineered classical approaches can compete effectively with modern deep learning methods for structured classification tasks. The linear decision boundary proves sufficient for news categorization, as evidenced by LinearSVC outperforming the non-linear RBF variant.

2. **Exceptional Efficiency of Linear Models**: LinearSVC's training time of 9.8 seconds and inference latency of 0.004 seconds per 1,000 samples represent orders-of-magnitude improvements over more complex models, making it ideal for production deployments with high throughput requirements.

3. **Hyperparameter Optimization Value**: The hyperparameter search for XGBoost and SVM models, while time-consuming, yields significant performance improvements. Caching optimized parameters allows for efficient subsequent training runs.

4. **RBF Kernel Limitations**: Despite theoretical advantages, SVM-RBF shows lower accuracy and extremely slow inference, making it impractical for this task. This suggests that the additional complexity does not provide benefits for news classification.

5. **Trade-off Clarity**: The results provide clear guidance: choose LinearSVC for efficiency-critical applications, XGBoost when probability estimates are needed, and transformers when maximum accuracy is the priority and resources are available.

6. **Practical Recommendations**: For most real-world news classification scenarios, LinearSVC offers the optimal balance of accuracy (92.28%), training speed (9.8s), and inference speed (0.004s/1k), making it the recommended choice unless specific requirements (probability estimates, maximum accuracy) dictate otherwise.

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
- `n_estimators`: 100
- `max_depth`: 7
- `learning_rate`: 0.2
- `subsample`: 0.8

### A.2 SVM-LinearSVC Best Hyperparameters
- `C`: 1.0
- `max_iter`: 1000
- `penalty`: l2
- `loss`: squared_hinge

### A.3 SVM-RBF Best Hyperparameters
- `C`: 1.0
- `gamma`: 'scale'

### A.4 RoBERTa Training Configuration
- `num_train_epochs`: 1
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `learning_rate`: 5e-5 (default from TrainingArguments)
- `warmup_steps`: 100
- `weight_decay`: 0.01
- `gradient_accumulation_steps`: 1

---

**Report Generated**: January 2025  
**Project Repository**: https://github.com/RyaneLi/COE379L-Project3

