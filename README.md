# NLP-Based-Cybercrime-Complaint-Classification

## Problem Statement

The goal of this project is to develop a robust Natural Language Processing (NLP) model capable of categorizing complaints based on parameters like the victim, type of fraud, and other relevant criteria. This project emphasizes the entire pipeline, from raw data preparation to the final classification and evaluation of the model.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
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



