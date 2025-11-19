"""
Project Proposal: Fake vs Real News Classifier

Project Overview:
The project is aim to develop a text classification model using natural language processing (NLP) to distinguish between fake and real news articles.
This model will be trained on a labeled dataset containing real and fake news articles.
The textual data will be transformed into numerical features suitable for machine learning algorithms.

Project Objectives:
1. **Data Collection and Preprocessing**:
Data collection: Kaggle Fake News Detection Datasets including two types of articles, fake news and true news. 
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets 
Preprocessing the raw dataset for model

2. **Model Development**:


3. **Model Evaluation and Optimization**:


4. **Visualization and Reporting**:


5. **Interactive Web App
Building a streamlit app with text input box and prediction output. 
For showcase and performance validation, browsering news which have already been identified as either false or true new, 
then inputting those news to the app and comparing the prediction output with the actual result.



Scope of Work:

**Phase 1: Requirements Gathering**
- Dateset identification
- Business goal identification
- FE design

**Phase 2: Data Collection and Preprocessing**
- Load data from Kaggle API
- Combine both fake dataset and true dataset, and create a label column. 
- Remove punctuation, special characters from the text statment.
- Converted all text to lowercase to reduce vocabulary size.
- Split sentences into individual tokens using nltk
- Remove stopwords which have little semantic meaning

**Phase 3: Model Development**
- Feature extraction/vectorization: using TF-IDF
- Data split into training set(70%), validation set(15%), test set(15%)
- Model selection (fake news detection is a binary classification)

**Phase 4: Model Evaluation and Optimization**
- Using appropriate performance metrics (accuracy, precision, F1 score)
-

**Phase 5: Visualization and Reporting**
- Visualize 

**Phase 6: Deployment: Interactive Web App**
-

**Phase 7: Testing and Validation**
- Browsering news 


Deliverables:

Timeline:


Risks and Mitigation:


Conclusion:
"""