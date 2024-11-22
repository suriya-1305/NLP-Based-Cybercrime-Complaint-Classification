# NLP-Based-Cybercrime-Complaint-Classification

## Problem Statement

The goal of this project is to develop a robust Natural Language Processing (NLP) model capable of categorizing complaints based on parameters like the victim, type of fraud, and other relevant criteria. This project emphasizes the entire pipeline, from raw data preparation to the final classification and evaluation of the model.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Approach](#approach)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)


## Project Objectives

The goal of this project is to create an end-to-end NLP solution for classifying unstructured text-based cybercrime complaints into relevant categories and subcategories. The objectives include:

- **Text Preprocessing**: Cleaning, tokenization, stemming, and stop word removal to convert raw text into a usable format.
- **Model Development**: Building and evaluating machine learning and NLP models to classify complaints with high accuracy.
- **Performance Evaluation**: Measuring model effectiveness using metrics like accuracy, precision, recall, and F1-score.
- **Deployment Readiness**: Delivering a scalable and interpretable solution for real-world use in cybercrime analysis.


## Project Overview

This project addresses the classification of a large, unstructured dataset of cybercrime complaints (~1.56 lakh records) into predefined categories and subcategories. These complaints cover a range of crimes, such as financial fraud, identity theft, and online abuse.

The project involves full control over the data pipeline, starting with **exploratory data analysis (EDA)** and **feature engineering**, through to **model development** and **performance evaluation**. 

### Key Challenges:
- Cleaning and processing messy text with errors, inconsistencies, and abbreviations.
- Addressing class imbalances across categories and subcategories.
- Ensuring model generalization across diverse datasets.

### Deliverables:
- A complete NLP pipeline for automated complaint classification.
- Saved models and embeddings for reuse and experimentation.
- Insights from exploratory data analysis.
- Detailed model performance reports with metrics and findings.

This repository is structured for ease of use, with clearly organized scripts, notebooks, and resources for reproducibility and further exploration.

## File Structure

```bash
NLP-Based-Cybercrime-Complaint-Classification/
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
├── data/
│   ├── raw/                 # Raw dataset
│   ├── processed/           # Cleaned and tokenized data
├── notebooks/               # EDA and experimental notebooks
│   ├── EDA.ipynb
│   ├── Model_Experiments.ipynb
├── assets/ 
│   ├── CategoryClassification/      # Saved models and embeddings
│   ├── SubCategoryClassification/   # Saved models and embeddings
├── results/                 # Results and reports
│   ├── performance.xlsx     # Model performance metrics
└── Final_script_with_category_and_SubCat_classification.ipynb
```

### **Assets for Subcategory Classification**
We are performing subcategory classification based on an initial split using the main category. This allows us to classify complaints into more specific subcategories within each main category. To facilitate this, the repository contains three main category folders:

- **category_0**: Women/Child Related Crime
- **category_1**: Financial Fraud Crimes
- **category_2**: Other Cyber Crimes

Inside each category folder, you will find the following assets:
- **Top-performing model weights**: These are the trained models that perform the best for each specific subcategory classification task within the category.
- **Embeddings**: These represent the feature vectors for the text data used in training the models, which are crucial for consistent prediction and model performance.

These assets are stored for reuse and experimentation, ensuring that each model can be applied to relevant complaints for efficient classification.

## Approach

We are performing subcategory classification based on an initial split using the main category. Below are the three main categories:

- **category_0**: Women/Child Related Crime
- **category_1**: Financial Fraud Crimes
- **category_2**: Other Cyber Crimes

### **Labels and their corresponding subcategory group in category 0:**
-	Label 2: Child Pornography/Child Sexual Abuse Material(CSAM)
-	Label 24: Rape/Gang Rape-Sexually Abusive Content
### **Labels and their corresponding subcategory group in category 1:**
-	Label 1: Business Email Compromise/Email Takeover
-	Label 3: Cryptocurrency Fraud
-	Label 8: Debit/Credit Card/Sim Swap Fraud
-	Label 9: Demat/Depository Fraud
-	Label 11: E-Wallet Related Frauds
-	Label 13: Fraud Call/Vishing
-	Label 16: Internet Banking-Related Fraud
-	Label 19: Online Gambling/Betting Fraud
-	Label 27: UPI-Related Frauds

### **Labels and their corresponding subcategory group in category 2:**
-	Label 0: Any Other Cyber Crime
-	Label 4: Cyber Bullying/Stalking/Sexting
-	Label 5: Cyber Terrorism
-	Label 6: Damage to Computer Systems
-	Label 7: Data Breach/Theft
-	Label 10: Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks
-	Label 12: Email Hacking/Phishing/Impersonating/Intimidating
-	Label 14: Hacking/Website Defacement
-	Label 15: Impersonation Fraud
-	Label 17: Malicious code attacks
-	Label 18: Online Cyber Trafficking
-	Label 20: Online Job Fraud
-	Label 21: Online Matrimonial Fraud
-	Label 22: Profile Hacking/Identity Theft
-	Label 23: Provocative Speech of Unlawful Acts
-	Label 26: Tampering with Computer Source Documents
-	Label 28: Unauthorized Access/Data Breach



## Setup instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/suriya-1305/NLP-Based-Cybercrime-Complaint-Classification.git
cd NLP-Based-Cybercrime-Complaint-Classification
```

### Step 2: Create a Virtual Environment

Set up a virtual environment to manage the dependencies:

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Step 3: Install Dependencies

Use `pip` to install the required libraries:

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

This code used NLTK stopwords, so ensure you download the necessary data:

```python
import nltk
nltk.download('stopwords')
```
### Step 5: Setup SpaCy:

Ensure SpaCy's model for English is downloaded

```bash
python -m spacy download en_core_web_sm
```

### Run the Notebook

launch Juypter Notebook to start working:
```bash
Juypter notebook
```

### Step 7: Optional: Verify Installation:

To confirm all the dependencies are installed correctly
```bash
pip list
```

## Usage

### Running the App

Use the provided notebooks for experimentation and testing various classification models. For example, use the `categoryClassification.ipynb` or `SubCategoryClassification.ipynb` notebook to compare the performance of different models on the dataset.

## Tech Stack
The project utilizes the following technologies:

- **Programming Language**: Python 3.x
- **Libraries**:
  - **pandas**: Data manipulation and analysis.
  - **NumPy**: Numerical operations.
  - **Matplotlib**: Data visualization.
  - **scikit-learn**: Machine learning models, preprocessing, and evaluation metrics.
  - **NLTK**: Natural Language Toolkit for text preprocessing.
  - **SpaCy**: Advanced NLP tools for tokenization, lemmatization, and named entity recognition.
  - **XGBoost**: Gradient boosting library for classification tasks.
  - **Imbalanced-learn**: Tools for handling class imbalance during model training.
  - **TQDM**: Progress bar for loops and operations.
- **Environment**: Jupyter Notebook or Python scripts.
- **Version Control**: Git for versioning and GitHub for repository hosting.

## Future Improvements

 1. **Contextual Embeddings**: Incorporate pretrained word embeddings like GloVe or Word2Vec to improve model understanding of the text.
 2. **Data correction using LLM before model training**: for eg. use Gemini to correct spelling mistakes, translate indigenous langs to English



