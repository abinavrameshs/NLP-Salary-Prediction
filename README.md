# NLP-Salary-Prediction
## Prediction of salary using text analysis

## Data Description

Please use the following link to download the data : 
http://www.kaggle.com/c/job-salary-prediction

Use training dataset train_rev1

## Project description
Predict salary from job description; the idea here is to test the predictive power of text and compare it with that of numeric data

## Methodology
Salary information was bucketized based on percentiles (high (75th percentile and above) or low (below 75th percentile) salary from the text contained in the job descriptions)

There are 3 classification models that were built for determining the salary range : 
1. Using only Text predictors : Two kinds of Naive Bayes models were built for this purpose 
  * Bernoulli Classifier
  * Multinomial Text Classifier
  
2. Using only numerical predictors

3. Hybrid model (using both numerical and text predictors)

The above models were then compared using accuracy score.

For detailed information on the project, please view the document attached in the repository.
