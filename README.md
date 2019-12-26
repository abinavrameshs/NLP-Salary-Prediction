# NLP-Salary-Prediction
## Prediction of salary using text analysis

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
