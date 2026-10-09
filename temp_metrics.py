import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score

base = pd.read_csv('data/Tweets.csv')
res = pd.read_csv('outputs/roberta_results.csv')
base['roberta'] = res['roberta'].values
base['roberta_score'] = res['roberta_score'].values
analyzer = SentimentIntensityAnalyzer()


def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def vader_label(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    return 'neutral'

base['clean_text'] = base['text'].apply(clean_tweet)
base['vader_clean'] = base['clean_text'].apply(vader_label)

for col in ['vader_clean', 'roberta']:
    acc = accuracy_score(base['airline_sentiment'], base[col])
    f1 = f1_score(base['airline_sentiment'], base[col], average='macro')
    print(f'{col}|acc={acc:.4f}|macro_f1={f1:.4f}')

neg = base[base['airline_sentiment'] == 'negative']
counts = neg['negativereason'].fillna('Unknown').str.strip().value_counts()
print('top_reasons')
print(counts.head(10).to_string())
print('share')
print((counts / counts.sum() * 100).round(2).head(5).to_string())
print('label_dist')
print(base['airline_sentiment'].value_counts(normalize=True).round(3).to_string())
print('roberta_dist')
print(base['roberta'].value_counts(normalize=True).round(3).to_string())
