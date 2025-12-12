Use of AI
---------

[1]. Tool: ChatGPT (via Cursor AI)
     Prompt: Create a Jupyter notebook structure for AG News dataset loading and initial exploratory data analysis with proper imports and data visualization setup
     Output:
       - Import statements for pandas, numpy, matplotlib, seaborn, datasets
       - Code snippet for loading AG News dataset from Hugging Face using load_dataset()
       - Basic data exploration code structure

[2]. Tool: ChatGPT (via Cursor AI)
     Prompt: Write Python code to combine title and description fields from AG News dataset into a single text input for classification
     Output:
       - Function to prepare text field (initially named combine_text_fields, later renamed to prepare_text_field)
       - Code structure for handling AG News dataset's text field

[3]. Tool: ChatGPT (via Cursor AI)
     Prompt: Create a comprehensive Jupyter notebook for classical machine learning models (XGBoost and SVM) with TF-IDF feature extraction, hyperparameter tuning using RandomizedSearchCV, and evaluation metrics including accuracy, macro F1-score, log loss, training time, and inference latency
     Output:
       - TF-IDF vectorization code with ngram_range, max_features, min_df, max_df parameters
       - XGBoost classifier initialization and RandomizedSearchCV setup
       - SVM (LinearSVC and SVC with RBF kernel) implementation code
       - Model evaluation function (evaluate_model) with metrics calculation
       - Inference latency measurement code structure
       - Results visualization code snippets

[4]. Tool: ChatGPT (via Cursor AI)
     Prompt: Create a Jupyter notebook for fine-tuning RoBERTa-base transformer model using Hugging Face Transformers library for text classification on AG News dataset, including training loop, evaluation metrics, inference latency measurement, and comparison with classical models
     Output:
       - RoBERTa model and tokenizer initialization code using AutoTokenizer and AutoModelForSequenceClassification
       - Hugging Face Trainer setup with TrainingArguments configuration
       - Custom TrainerCallback class (ProgressCallback) for training progress tracking
       - compute_metrics function for evaluation
       - Inference latency measurement code
       - Results comparison code structure

[5]. Tool: ChatGPT (via Cursor AI)
     Prompt: Fix TrainingArguments parameter error - evaluation_strategy should be eval_strategy in newer transformers version
     Output:
       - Changed evaluation_strategy="steps" to eval_strategy="steps" in TrainingArguments

[6]. Tool: ChatGPT (via Cursor AI)
     Prompt: Fix TrainingArguments error - save_strategy must match eval_strategy when load_best_model_at_end=True
     Output:
       - Changed save_strategy from "epoch" to "steps" and added save_steps=500 to match eval_steps

[7]. Tool: ChatGPT (via Cursor AI)
     Prompt: Add progress tracking and logging for training and hyperparameter search in notebooks
     Output:
       - ProgressTracker class for RandomizedSearchCV progress tracking
       - Thread-based progress pinger for LinearSVC and RBF SVC training
       - Enhanced ProgressCallback for Hugging Face Trainer with detailed logging
       - Progress logging code for evaluation and inference latency measurement

[8]. Tool: ChatGPT (via Cursor AI)
     Prompt: Add parameter caching to skip hyperparameter search on subsequent runs
     Output:
       - CACHED_XGB_BEST_PARAMS and CACHED_SVM_RBF_BEST_PARAMS dictionaries
       - USE_CACHED_XGB_PARAMS and USE_CACHED_SVM_RBF_PARAMS flags
       - Conditional logic to use cached parameters instead of running search

[9]. Tool: ChatGPT (via Cursor AI)
     Prompt: Fix sparse matrix length error - use shape[0] instead of len() for sparse matrices
     Output:
       - Replaced len(X_train_tfidf) with X_train_tfidf.shape[0] in multiple locations
       - Updated evaluate_model function to handle sparse matrices correctly

[10]. Tool: ChatGPT (via Cursor AI)
      Prompt: Disable MPS backend and force CPU usage for PyTorch to avoid M1 GPU memory issues
      Output:
        - Code to disable MPS: torch.backends.mps.is_available = lambda: False
        - Environment variable setting: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        - Explicit CPU device assignment code

[11]. Tool: ChatGPT (via Cursor AI)
      Prompt: Reduce training data subset and epochs to complete training in under 1 hour
      Output:
        - Stratified sampling code using train_test_split to create balanced subset
        - TRAIN_SUBSET_SIZE variable set to 1200 samples
        - Training arguments adjusted: num_train_epochs=1, warmup_steps=100, eval_steps=200
