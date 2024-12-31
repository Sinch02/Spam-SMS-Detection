# Spam-SMS-Detection
A machine learning model to detect spam SMS messages

Spam SMS Detection
This project builds a machine learning model to classify SMS messages as either 'spam' or 'ham' (non-spam). The model is trained and evaluated using various machine learning algorithms.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

# Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Authors](#authors)

Project Overview
The goal is to create a model that can tell if an SMS is spam or not. Here's the process:

1. Preprocess the text (clean up data and convert it to numbers using TF-IDF).
2. Train models using:
   - Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
3. Evaluate models using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

Features
- Label: "ham" (non-spam) or "spam"
- Message: Content of the SMS

Dataset
The dataset consists of labeled SMS messages (spam or ham) and can be downloaded from:

- [Spam SMS Dataset](URL-to-dataset)

Tech Stack
- Python: For coding
- Jupyter Notebook: For running code interactively
- Pandas: For data manipulation
- NumPy: For numerical calculations
- Scikit-learn: For machine learning
- NLTK: For text preprocessing
- Matplotlib/Seaborn: For visualizing results (optional)

Installation
Clone the repository:

```bash
git clone <repository_url>
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Open the Jupyter Notebook:

```bash
jupyter notebook Spam_SMS_Detection.ipynb
```

Project Workflow
1. **Data Loading and Preprocessing:**
   - Load and clean the dataset.
   - Remove unnecessary characters and tokenize the text.

2. **Feature Extraction:**
   - Convert text data into numbers using TF-IDF.

3. **Model Training:**
   - Train models using Naive Bayes, Logistic Regression, and SVM.

4. **Model Evaluation:**
   - Evaluate models based on accuracy, precision, recall, and F1-score.

Results
### Naive Bayes:
- Accuracy: 97.40%
- Precision: 1.00
- Recall: 0.81
- F1-Score: 0.89

### Logistic Regression:
- Accuracy: 95.07%
- Precision: 0.95
- Recall: 0.67
- F1-Score: 0.78

### SVM:
- Accuracy: 97.94%
- Precision: 0.98
- Recall: 0.87
- F1-Score: 0.92

**Key Insight:** SVM performed the best with 97.94% accuracy.

How to Run the Project
1. Install required libraries.
2. Open the Spam_SMS_Detection.ipynb notebook.
3. Run the cells to load the data, preprocess it, train models, and evaluate them.

Future Improvements
- Try advanced models like XGBoost or Neural Networks.
- Handle class imbalance with oversampling techniques.
- Fine-tune hyperparameters for better performance.
- Add more text preprocessing steps (e.g., lemmatization, bigrams).

License
This project is licensed under the MIT License. See the LICENSE file for details.

Authors
- Sinchana B R
  - Contact: sinchanabr02@gmail.com

Feel free to reach out with any questions or suggestions!
