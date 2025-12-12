# COE379L Project 3: Cross-Model Comparison for News Topic Classification

## Project Overview

This project performs a comprehensive comparison between classical machine learning algorithms (XGBoost, SVM with TF-IDF) and fine-tuned transformer models (RoBERTa) for news topic classification using the AG News dataset.

## Project Structure

```
COE379L-Project3/
├── 01_Data_Preprocessing_and_EDA.ipynb  # Data loading, preprocessing, and EDA
├── Project_Requirements.txt              # Detailed project requirements
├── Use_of_AI.md                          # AI tool usage documentation
├── COE379L-Project03-Initial-Proposal.pdf
├── README.md                             # This file
├── .gitignore
├── venv/                                 # Virtual environment
└── data/                                 # Processed data files (gitignored)
```

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install required packages:**
   ```bash
   pip install pandas numpy matplotlib seaborn datasets transformers torch scikit-learn xgboost
   ```

3. **Run notebooks:**
   - Start with `01_Data_Preprocessing_and_EDA.ipynb` to load and preprocess the AG News dataset
   - Additional notebooks for classical models and transformer fine-tuning will be added

## Dataset

- **Source:** Hugging Face Datasets Hub (`ag_news`)
- **Training samples:** 120,000+
- **Test samples:** 7,600
- **Classes:** 4 balanced categories (World, Sports, Business, Sci/Tech)

## Deliverables

1. Jupyter Notebooks:
   - Data Preprocessing and EDA
   - Classical Model Implementation (XGBoost, SVM)
   - Transformer Fine-Tuning (RoBERTa)

2. Final Technical Report (2 pages)
   - Methodology
   - Quantitative results table
   - Performance vs. efficiency analysis

3. Visualizations
   - Confusion matrices
   - Performance comparison charts

## AI Usage

This project uses AI tools for certain technical portions. All AI-generated code is documented in `Use_of_AI.md` with corresponding comments in the notebooks.

## License

This is an academic project for COE379L.

