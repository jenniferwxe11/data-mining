# Insurance Fraud Detection Using Machine Learning Models

## 1. Introduction

### 1.1. Problem Statement & Motivation

Insurance fraud has seen a significant rise in recent years, particularly in Singapore, where cases have more than tripled from 20 in 2018 to 71 in 2021. The highest recorded motor insurance scam in 2018 amounted to $1.6 million (Magbanua, 2022). The current methods used by the insurance industry to detect fraud are inadequate, prompting the need for more efficient models.

Insurance fraud is not only a problem for insurance companies but also for consumers. Financial losses due to fraud impact the companies' ability to pay legitimate claims, which can result in higher premiums for consumers. In the US alone, insurance fraud costs an estimated $310 billion annually (insurancefraud.org, 2022).

In the motor insurance industry, fraud detection is especially time-consuming. Victims of accidents are often pressured by scammers to use certain workshops or clinics that inflate repair or medical costs. Insurance companies then need to manually review each claim, a resource-intensive process. By implementing trained models to reinforce fraud detection systems, we can achieve a more optimal method of detecting insurance fraud.

## 2. Literature Review

### 2.1 Detecting Insurance Fraud with Machine Learning

Machine learning models have been developed to address insurance fraud. A widely used technique is deep anomaly detection, which analyzes genuine claims to form a model that can identify anomalies in larger datasets (Markovskaia, 2022). Predictive analytics builds on anomaly detection by analyzing features of anomalous claims to reduce overall workload. However, previous models have used small sample sizes, limiting their conclusions. Additionally, ensemble methods like Bagging and Boosting have not been extensively used in fraud detection, indicating the need for further analysis.

### 2.2 Fraud Detection and Analysis for Insurance Claims Using Machine Learning

Fraudulent insurance claims are rare, accounting for only 1% of all claims (A. Urunkar, Khot, Bhat, & Mudegol, 2022), creating an imbalanced dataset that leads to biased predictions. We employed the Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic examples of the minority class and create a more balanced dataset, thus reducing model bias. Research indicates that Random Forest has higher prediction accuracy while Logistic Regression performs well with small datasets, guiding our choice of models for this project.

### 2.3 Comparative Analysis of Machine Learning Techniques for Detecting Insurance Claims Fraud

Model performance can also be evaluated using Receiver Operating Characteristics (ROC) curves (Guha, 2018). The closer the curve is to the Y-axis, the better the model’s performance. We will use ROC curves with the area under the curve (AUC) to compare different models' performances.

## 3. Dataset

### 3.1 Fraud Motor Insurance Dataset

We used the insurance_claims.csv dataset from Kaggle, which contains information about automotive insurance claims, including policy details, claimant information, incident details, and whether fraud was reported. The dataset can be accessed [here](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data).

## 4. Methodology

### 4.1 Data Pre-processing

#### 4.1.1 Categorical Encoding

We performed label encoding for categorical values in the dataset, such as auto_make and auto_model, as models do not understand categorical values. Label encoding was chosen over one-hot encoding due to the ordered nature of the variables and to avoid creating numerous columns.

#### 4.1.2 Feature Selection

We dropped unnecessary features like incident_dates and policy_numbers, created a correlation matrix to select features with a correlation threshold of 0.05, and checked for multicollinearity using the variance inflation factor (VIF). The selected features were:

- number_of_vehicles_involved (Discrete)
- umbrella_limit (Continuous)
- total_claim_amount (Continuous)
- incident_type_encode (Numeric categorical)
- collision_type_encode (Numeric categorical)
- incident_severity_encode (Numeric categorical)
- incident_state_encode (Numeric categorical)

#### 4.1.3 Scaling

We scaled the data to ensure models were not affected by variable ranges, using `.fit_transform` for the training set and `.transform` for the testing set (Khanna, 2020).

### 4.2 Model Building

#### 4.2.1 Decision Tree & Random Forest

We started with Decision Tree models but found them lacking, leading us to try Random Forest to add randomness and achieve better results.

#### 4.2.2 Logistic Regression

Logistic regression was chosen for its effectiveness in classification problems. We faced issues with boosting, which were resolved by hyperparameter tuning using grid search, leading to better performance.

#### 4.2.3 k-Nearest Neighbors (kNN)

We found that k=9 provided the best accuracy for kNN. Bagging was applied but boosting was not used due to kNN's unsuitability for the AdaBoost framework.

#### 4.2.4 Naive Bayes

We tested Gaussian, Multinomial, and Bernoulli Naive Bayes models, with Bernoulli performing best. GridSearchCV was used for hyperparameter tuning. One-hot encoding improved the Bernoulli model's accuracy.

#### 4.2.5 Neural Network

A simple neural network with 7 input nodes, 3 dense layers, and binary cross-entropy loss was used. The network was not fully optimized due to the team's limited expertise.

### 4.3 Data Generation

SMOTE was used to balance the dataset, training models equally on both outcomes. SMOTE was applied only to the training set to avoid unrealistic test set scenarios.

## 5. Results and Discussion

### 5.2 Logistic Regression

Hyperparameter tuning improved the model's F1 score significantly. Without SMOTE, the tuned model performed best, while with SMOTE, the base and bagging models performed equally well.

### 5.3 k-Nearest Neighbor

kNN showed decent accuracy but low precision, recall, and F1 scores. Results were consistent even with SMOTE, indicating kNN's unsuitability for this project.

### 5.4 Decision Tree & Random Forest

While performing well on training data, Decision Tree and Random Forest did not significantly improve testing data results. Thus, they are not suitable for this classification problem.

### 5.5 Naive Bayes

Bernoulli Naive Bayes performed best among the tested models. One-hot encoding further improved its accuracy. Bagging and boosting did not enhance performance, suggesting the model had already achieved optimal performance.

### Summary of Findings

Ensemble methods like Bagging and Boosting can improve model performance but must be carefully applied. SMOTE effectively balances datasets but may cause overfitting. Logistic Regression with hyperparameter tuning and Bernoulli Naive Bayes with one-hot encoding showed the best results.

## Conclusion

Fraud detection in the insurance industry can be significantly improved using machine learning models. Logistic Regression and Bernoulli Naive Bayes emerged as the best-performing models in our study, with careful data pre-processing and hyperparameter tuning being crucial for optimal performance. Further research and more extensive datasets could enhance these models and provide even more robust fraud detection solutions.
