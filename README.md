# Sentiment-Analysis-on-Amazon-Product-Review-


This project analyzes Amazon Fine Food customer reviews and builds machine-learning models to classify review sentiment from review text. It contains two Jupyter notebooks: one focused on exploratory data analysis and text representations, and another focused on supervised sentiment classification and model comparison.


Project Notebooks
- NLP_EDA_TextP.ipynb — Data cleaning, exploratory data analysis (EDA), text preprocessing, TF-IDF features, Word2Vec embeddings, n-gram frequency analysis, and review/product/user-level visualizations.
- NLP_Modeling.ipynb — Preprocessing for classification, binary sentiment-label creation, TF-IDF feature extraction, training and evaluation of Logistic Regression, Multinomial Naive Bayes, and Linear SVM, followed by example predictions.


Dataset
Both notebooks expect a CSV file named Reviews.csv in the notebook's working directory.

The notebooks use fields including:
- ProductId, UserId, ProfileName
- Score, Summary, Text
- HelpfulnessNumerator, HelpfulnessDenominator
- Time

Dataset link -  https://www.kaggle.com/datasets/snap/...


Workflow


1. Data cleaning and feature creation

The notebooks sort reviews by product, remove duplicates based on UserId, ProfileName, Time, and Text, filter records where helpfulness votes are inconsistent (HelpfulnessNumerator > HelpfulnessDenominator), and remove rows missing ProfileName or Summary.

Additional fields include review length, helpfulness ratio, review date, year, month, and year-month. The modeling notebook also creates a cleaned text field.


2. Text analysis and representation
   
The exploratory notebook includes:
- Text normalization, including URL and HTML removal, contraction expansion, lowercasing, and removal of non-alphabetic characters.
- TF-IDF vectorization with unigrams and bigrams (ngram_range=(1, 2), min_df=10).
- Word2Vec word embeddings (50-dimensional vectors, min_count=5) and average / TF-IDF-weighted review vectors.
- Unigram and bigram frequency analysis for positive and negative reviews.
- Rating and sentiment distributions, word clouds, review-length analysis, helpfulness analysis, time trends, product/user summaries, and correlation analysis.
In this notebook, ratings are grouped into three sentiment categories: scores 1–2 are Negative, score 3 is Neutral, and scores 4–5 are Positive.
3. Binary sentiment classification (NLP_Modeling.ipynb)
For the classification task, the notebook uses a binary target:
- 0 — Negative: rating score 1 or 2
- 1 — Positive: rating score 3, 4, or 5
This is the notebook's implemented labeling rule; therefore, neutral three-star reviews are included in the positive class for modeling.

The data is split into training and test sets using an 80/20 split, random_state=42, and stratification. TF-IDF features use unigrams and bigrams with min_df=10. The models are evaluated using accuracy, macro F1, classification reports, and a confusion matrix where provided.


Model Results


The following results are recorded in the saved output of NLP_Modeling.ipynb:
Model	Accuracy	Macro F1
Logistic Regression (C=1, max_iter=1000, class_weight='balanced')	0.881762	0.798090
Multinomial Naive Bayes	0.869397	0.568935
Linear SVM (random_state=42)	0.892324	0.801229


On the saved test-set results, Linear SVM performs best by both accuracy and macro F1.



The machine learning model has been successfully deployed and is accessible through the following link:

Live Application: https://sentiment-analysis-on-amazon-product-9txh.onrender.com



Requirements


The notebooks use Python and common data-science/NLP libraries, including:
- pandas, numpy
- matplotlib, seaborn, plotly, cufflinks
- nltk, textblob, wordcloud
- beautifulsoup4 (BeautifulSoup), lxml
- scikit-learn
- gensim
- tqdm
The notebooks also download NLTK resources such as English stopwords and tokenizer data.
