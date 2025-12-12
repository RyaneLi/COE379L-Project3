# Video Script: Cross-Model Comparison for News Topic Classification
**Duration: 10 minutes**  
**COE379L Project 3**

---

## [0:00 - 0:30] Introduction and Project Overview

**[Slide 1: Title Slide]**

"Hello, I'm [Your Name], and today I'll be presenting my project on Cross-Model Comparison for News Topic Classification. This project compares classical machine learning approaches with modern transformer-based models for text classification tasks."

**[Slide 2: Project Objective]**

"The primary objective of this project is to conduct a comprehensive, direct comparison between two distinct methodological paradigms for multi-class text classification. On one hand, we have classical feature-engineered models using XGBoost and Support Vector Machines with TF-IDF. On the other hand, we have fine-tuned transformer models, specifically RoBERTa-base."

"The goal is to determine the optimal modeling approach by systematically evaluating both computational efficiency and predictive performance on a standardized news classification task."

---

## [0:30 - 1:30] Research Questions and Dataset

**[Slide 3: Research Questions]**

"This study addresses three key research questions: First, how significant is the performance gap between classical and transformer models on a modern news classification task? Second, what are the trade-offs in training time and inference latency for each approach? And third, can a resource-efficient classical model provide sufficient performance to justify avoiding the higher computational costs of fine-tuning large transformers?"

**[Slide 4: AG News Dataset]**

"For this comparison, we used the AG News dataset from Hugging Face, which contains over 120,000 training samples and 7,600 test samples. The dataset consists of four balanced categories: World, Sports, Business, and Sci/Tech. Each sample contains a news article title and description, which we combined into a single text input for classification consistency across all models."

"The balanced nature of this dataset makes macro-averaged metrics appropriate for evaluation, as they treat all classes equally."

---

## [1:30 - 3:30] Methodology: Classical Models

**[Slide 5: Classical Models Pipeline]**

"Let me now explain our methodology, starting with the classical models pipeline."

"For feature extraction, we used Term Frequency-Inverse Document Frequency, or TF-IDF, which captures term importance within documents relative to the corpus. We configured it to use both unigrams and bigrams to capture word-level and phrase-level patterns, with a vocabulary limited to the top 50,000 features for computational efficiency."

**[Slide 6: XGBoost Model]**

"For our first classical model, we implemented XGBoost, an ensemble gradient boosting method known for strong performance on text classification. We used RandomizedSearchCV with 2-fold cross-validation to optimize hyperparameters including the number of estimators, learning rate, tree depth, and subsample rate. The best parameters we found were: 100 estimators, max depth of 7, learning rate of 0.2, and subsample of 0.8."

**[Slide 7: SVM Models]**

"For our second approach, we implemented two variants of Support Vector Machines. First, LinearSVC, which is efficient for large sparse matrices. We optimized it using RandomizedSearchCV and found the best parameters to be C equals 1.0 and max_iter of 1000. Second, we tested SVC with an RBF kernel, which can capture non-linear decision boundaries but is computationally expensive. The best RBF parameters were C equals 1.0 and gamma set to 'scale'."

"Due to computational constraints, the RBF SVM was trained on a smaller subset of 2,000 samples."

---

## [3:30 - 5:00] Methodology: Transformer Models

**[Slide 8: RoBERTa-base Model]**

"Moving to the transformer pipeline, we selected RoBERTa-base, which stands for Robustly Optimized BERT Pretraining Approach. We chose RoBERTa over BERT because it uses dynamic masking, larger batch sizes, and typically yields superior results on downstream classification tasks."

**[Slide 9: Fine-Tuning Process]**

"For fine-tuning, we loaded the pre-trained RoBERTa-base model from Hugging Face and added a classification head mapping to our 4 classes. We fine-tuned the entire network end-to-end on the AG News dataset. Due to computational constraints on CPU, we used a subset of 1,200 training samples and trained for 1 epoch with a batch size of 16."

"Key training parameters included: learning rate of 5e-5, 100 warmup steps, weight decay of 0.01, and evaluation every 200 steps. The model was trained on CPU to ensure stability and avoid memory issues."

"Unlike TF-IDF's static word representations, RoBERTa generates contextual embeddings where word meanings adapt based on surrounding context, leveraging knowledge from pre-training on large text corpora."

---

## [5:00 - 7:00] Results and Performance Comparison

**[Slide 10: Performance Metrics Table]**

"Now let's examine the results. All models were evaluated using consistent metrics on the same test set: accuracy, macro-averaged F1-score, and log loss for performance, plus training time and inference latency for efficiency."

**[Note: Insert actual results table when available]**

"From our experiments, we observed that..."

**[If results are available, discuss them here. Otherwise, use placeholder structure:]**

"The transformer model, RoBERTa-base, achieved the highest accuracy and F1-score, demonstrating the power of contextual embeddings and pre-trained knowledge. However, this came at a significant computational cost."

"The classical models, particularly XGBoost and LinearSVC, showed competitive performance with much faster training and inference times. XGBoost provided a good balance between accuracy and efficiency, while LinearSVC was the fastest for both training and inference."

"The RBF SVM, despite its non-linear capabilities, showed similar performance to LinearSVC but with much higher computational costs, making it less practical for this task."

---

## [7:00 - 8:30] Efficiency Analysis and Trade-offs

**[Slide 11: Training Time Comparison]**

"Looking at computational efficiency, the differences are substantial. Classical models trained in minutes, with LinearSVC being the fastest at just a few minutes. XGBoost took slightly longer due to its ensemble nature, but still completed in under an hour."

"In contrast, the RoBERTa fine-tuning, even with our reduced subset, took significantly longer on CPU. This highlights the trade-off between model complexity and training time."

**[Slide 12: Inference Latency]**

"For inference latency, classical models excel with very fast prediction times, making them suitable for real-time applications. The transformer model, while accurate, has slower inference due to its size and complexity."

"This creates a clear trade-off: transformers offer superior accuracy but require substantial computational resources, while classical models provide good performance with much lower resource requirements."

---

## [8:30 - 9:30] Key Findings and Practical Recommendations

**[Slide 13: Key Findings]**

"Our key findings reveal that: First, while transformers achieve higher accuracy, the performance gap may not always justify the computational cost, especially for well-structured classification tasks like news categorization."

"Second, classical models, particularly XGBoost, provide an excellent balance between performance and efficiency, achieving competitive results with significantly lower resource requirements."

"Third, the choice between approaches depends heavily on the specific use case, available resources, and accuracy requirements."

**[Slide 14: Recommendations]**

"Based on our analysis, here are our practical recommendations:"

"For high-accuracy requirements with sufficient computational resources, RoBERTa-base is the clear choice. For real-time applications with moderate accuracy needs, XGBoost or LinearSVC are excellent options. For resource-constrained environments, LinearSVC provides the best efficiency. And for a balanced approach, XGBoost offers good accuracy with reasonable computational costs."

---

## [9:30 - 10:00] Conclusion and Future Work

**[Slide 15: Conclusion]**

"In conclusion, this project demonstrates that both classical and transformer approaches have their place in text classification. The choice depends on the specific requirements of accuracy, computational resources, and deployment constraints."

"Classical models remain highly relevant, especially when computational efficiency is a priority, while transformers excel when maximum accuracy is required and resources are available."

**[Slide 16: Future Work and Acknowledgments]**

"Future work could explore evaluation on additional datasets, comparison with other transformer architectures like DistilBERT for efficiency, ensemble methods combining classical and transformer models, and model compression techniques for transformer efficiency."

"Thank you for your attention. I'm happy to answer any questions."

---

## Presentation Tips:

1. **Timing**: Practice to ensure you stay within 10 minutes. Aim for ~1 minute per major section.

2. **Visuals**: 
   - Use slides with clear, readable text
   - Include visualizations (confusion matrices, comparison charts)
   - Show code snippets or architecture diagrams where helpful

3. **Delivery**:
   - Speak clearly and at a moderate pace
   - Pause briefly between sections
   - Use gestures to emphasize key points
   - Maintain eye contact with the camera

4. **Slides to Prepare**:
   - Title slide
   - Project objective
   - Research questions
   - Dataset overview
   - Methodology (classical models)
   - Methodology (transformer models)
   - Results table
   - Performance comparison charts
   - Training time comparison
   - Key findings
   - Recommendations
   - Conclusion
   - Future work

5. **Recording Setup**:
   - Use Zoom's "Record to the cloud" option
   - Ensure good lighting and audio quality
   - Share your screen to show slides
   - Have the script visible but don't read directly from it

---

**Total Estimated Words: ~1,200 words**  
**Speaking Time: ~10 minutes at normal pace**

