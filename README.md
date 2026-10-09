# Sentiment Analysis and Data Insights

This project explores airline-related tweet sentiment using a combination of rule-based and transformer-based approaches. The analysis compares VADER sentiment scoring with a RoBERTa model and prepares a cleaned dataset for dashboard reporting.

## Project goal

The main objective was to classify each tweet into one of the airline sentiment labels:

- negative
- neutral
- positive

and then evaluate how well the rule-based VADER method and the RoBERTa model match the airline-provided labels in the source data.

## Data source

The project reads the tweet dataset from [data/Tweets.csv](data/Tweets.csv). The data includes tweet text, airline, sentiment label, confidence, negative reason, and timestamp fields.

## Workflow

### 1. Data loading and cleanup

The notebook loads the dataset and keeps only the columns needed for analysis and export:

- tweet_id
- airline
- airline_sentiment
- airline_sentiment_confidence
- negativereason
- text
- tweet_created

The project also inspects the label distribution and reviews the first rows to validate the dataset structure.

### 2. Text preprocessing for VADER

A custom cleaning function removes URLs, @mentions, HTML leftovers, and repeated whitespace from tweet text before sentiment inference. This creates both a raw version and a cleaned version of each tweet.

The VADER sentiment analyzer is then applied to:

- the original text
- the cleaned text

This produces a sentiment label for each tweet using the standard VADER thresholds:

- positive if compound score >= 0.05
- negative if compound score <= -0.05
- neutral otherwise

### 3. Model evaluation and comparison

The notebook compares the airline labels against the VADER outputs using:

- accuracy
- classification report
- confusion matrix

This helps identify where VADER performs well and where it struggles, especially around neutral and negative sentiment interpretation.

### 4. RoBERTa sentiment prediction

The project uses a RoBERTa model from Hugging Face:

- cardiffnlp/twitter-roberta-base-sentiment-latest

The script in [score_roberta.py](score_roberta.py) prepares the text, runs batched sentiment inference, and saves the predictions to [outputs/roberta_results.csv](outputs/roberta_results.csv).

### 5. Cross-model comparison

Once the RoBERTa predictions are joined back to the tweet dataset, the notebook evaluates:

- VADER cleaned accuracy
- RoBERTa accuracy
- macro F1 scores
- classification reports

The analysis also checks a high-confidence subset where airline sentiment confidence is at least 0.7 to understand whether stronger labels result in cleaner model performance.

### 6. Duplicate and data quality checks

The project checks the dataset for duplicate tweet IDs and rows. It quantifies:

- repeated tweet IDs
- identical rows across all columns
- conflicting labels across duplicate IDs

After this review, it removes duplicate records before the final comparison metrics.

### 7. Dashboard-ready export

The final notebook creates a flattened export file at [outputs/dashboard_data.csv](outputs/dashboard_data.csv) with:

- tweet_id
- airline
- airline_sentiment
- airline_sentiment_confidence
- negativereason
- tweet_created
- text
- vader_clean
- roberta
- roberta_score
- date
- vader_correct flag
- roberta_correct flag

This output is structured for dashboard reporting and easy performance calculation.

## Project files

- [notebooks/01_explore.ipynb](notebooks/01_explore.ipynb) - main exploratory analysis and evaluation notebook
- [score_roberta.py](score_roberta.py) - RoBERTa inference pipeline
- [data/Tweets.csv](data/Tweets.csv) - source tweet dataset
- [outputs/roberta_results.csv](outputs/roberta_results.csv) - model output predictions
- [outputs/dashboard_data.csv](outputs/dashboard_data.csv) - dashboard-ready export
- [dashboard/airline_sentiment_dashboard.pbix](dashboard/airline_sentiment_dashboard.pbix) - dashboard artifact

## Key takeaway

This project demonstrates a practical sentiment-analysis workflow for social media data: preprocessing text, comparing a lightweight lexicon model with a transformer model, validating against labeled ground truth, and exporting a clean dataset ready for insight generation.

The notebook acts as the primary record of the analysis, while the Python script automates the RoBERTa prediction stage and the outputs folder stores the reusable result files.
