# Week 3 Insights Report: Airline Sentiment Analysis

## Executive Summary

The project examines public tweet sentiment for six US airlines using a two-model comparison: a rule-based baseline (VADER) and a transformer model (RoBERTa). It uses the Kaggle "Twitter US Airline Sentiment" dataset, which contains 14,640 tweets from 17-24 February 2015 and human-annotated labels supplied with the dataset. The analysis is an indicator of how dissatisfied customers talk about service, rather than a direct measure of service quality itself.

The label distribution is strongly negative: 9,178 negative tweets (62.7%), 3,099 neutral tweets (21.2%), and 2,363 positive tweets (16.1%). This distribution matters because customer complaints are concentrated among a relatively small set of operational issues. The analysis therefore turns social media text into operational insight by surfacing the complaint themes that drive the negative sentiment mix.

The main negative themes are customer service, late flights, and cancelled flights. In the negative subset, customer service accounts for 2,910 tweets (31.7%), and late flights account for 1,665 tweets (18.1%). Together, those two categories account for 49.8% of all negative tweets. This suggests that a large share of dissatisfaction is driven by how airlines handle service failures and disruption, not simply by flight availability or ticket pricing.

## Model Comparison Breakdown

The notebook compares two sentiment approaches:

- VADER, a general lexicon-based sentiment model
- RoBERTa, a transformer-based language model trained on tweets

The workflow in [notebooks/01_explore.ipynb](notebooks/01_explore.ipynb) evaluates both against the human-annotated labels supplied with the dataset using accuracy, macro F1, and classification reports. This comparison is not perfectly level because RoBERTa was trained on tweets while VADER is a general lexicon, so the transformer model is expected to perform better on social media language.

### Results

| Model | Accuracy | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| VADER (raw text) | 0.490 | — | Baseline result |
| VADER (cleaned text) | 0.550 | 0.514 | Cleaned text result |
| RoBERTa | 0.769 | 0.734 | Transformer model |
| Confidence subset (>= 0.7) | VADER 0.592 / RoBERTa 0.840 | VADER 0.532 / RoBERTa 0.790 | Robustness check |

RoBERTa outperformed VADER by about 22 points of accuracy on the headline result, and by about 25 points in the high-confidence subset. The contrast is meaningful because the transformer model is better suited to tweet language, while the rule-based system is more limited in context and domain nuance.

### VADER

VADER is useful as a baseline because it is fast and interpretable, and it is not designed specifically for airline tweet language. It achieves 0.49 accuracy on raw text and 0.550 accuracy on cleaned text, with 0.514 macro F1 on cleaned text. This shows that cleaning the text helps, but the rule-based model still underperforms a contextual transformer model on this task.

### RoBERTa

RoBERTa is the stronger contextual model in this project because it is a transformer classifier trained on tweets. It reaches 0.769 accuracy and 0.734 macro F1 on the full dataset, and 0.840 accuracy and 0.790 macro F1 in the high-confidence subset. This makes it the more reliable choice for airline tweet sentiment classification in this dataset.

## Sentiment by airline

The negative share by airline is as follows:

| Airline | Negative share |
| --- | ---: |
| US Airways | 77.7% |
| American | 71.0% |
| United | 68.9% |
| Southwest | 49.0% |
| Delta | 43.0% |
| Virgin America | 35.9% |

Negative sentiment is highest for US Airways (77.7%), American (71.0%) and United (68.9%), which are also the three airlines with the most tweets in the dataset (United 3,822, US Airways 2,913, American 2,759). Tweet volume reflects how many tweets were collected, not airline size, so this should not be read as an airline-size effect. Customer service is the top complaint for five of the six airlines. Delta is the exception, where late flights come first. Virgin America has only 504 tweets (181 negative), so its rate should be interpreted cautiously.

## Top Customer Complaint Drivers

The analysis shows that the main negative complaint categories are concentrated in a few operational issues. Among the 9,178 negative tweets:

- Customer service issue: 2,910 (31.7%)
- Late flight: 1,665 (18.1%)
- Can't Tell: 1,190 (13.0%)
- Cancelled flight: 847
- Lost luggage: 724
- Damaged luggage: 74
- Bad flight: 580

The three leading real complaint categories are customer service, late flights, and cancelled flights. Customer service + late flight = 49.8% of the negative tweets. Luggage complaints, when combined, total 798 tweets (724 lost luggage + 74 damaged luggage), which places luggage behind the top three real complaint categories. "Can't Tell" is not a real complaint category; it reflects uncertainty in the label assignment rather than a clear operational cause.

[TODO: add 3-4 example tweets from the error analysis]

## Strategic Operational Recommendations for Airlines

### 1. Prioritize customer service recovery for the highest-volume complaint category

Customer service is the most common negative complaint category, accounting for 2,910 of 9,178 negative tweets (31.7%). Airlines should focus on service recovery for slow replies, poor communication, and unresolved complaints, because this is the largest single driver of negative social sentiment.

### 2. Reduce delay-related dissatisfaction by improving operational transparency

Late flight issues account for 1,665 negative tweets (18.1%), and together with customer service they make up 49.8% of negative tweet volume. Operational teams should prioritize clearer disruption messaging and faster recovery actions when delays occur, because delay-related friction is a major contributor to the negative sentiment mix.

### 3. Treat cancelled flights as a service recovery priority

Cancelled flights account for 847 negative tweets. This is not the top category, but it is still a highly visible operational issue that strongly affects customer perception. Airlines should improve the way cancellations are communicated and recovered to reduce customer frustration after service disruption.

### 4. Manage luggage complaints as a visible customer experience issue

Lost luggage and damaged luggage together account for 798 negative tweets. This is a meaningful operational issue and should be tracked as part of the airline's service experience, especially where the customer experience breaks down after travel disruption.

### 5. Use the sentiment model as a reporting baseline, not a complete decision system

This dataset covers about eight days of US tweets (17 to 24 February 2015). It includes timestamps, but daily tweet volume varies widely (from 953 tweets on 17 February to 3,515 on 23 February), which makes day-to-day trends hard to interpret. It also includes self-reported location and time zone fields, but these are free text and missing for about a third of tweets, so regional analysis was not attempted. Future work should use a longer collection period and cleaned location data before treating sentiment as an operational early-warning system.

## Error analysis

The models struggle most with tweets that are sarcastic, polite, or emotionally muted. This is especially common when a customer describes a situation without clear emotional wording, such as waiting 26 minutes on hold. In those cases, the text may describe the problem clearly but not carry obvious sentiment words, which makes classification harder for both VADER and RoBERTa.

[TODO: add 3-4 real example tweets with human label, VADER label, RoBERTa label]

## Limitations

- The sample covers about eight days of US tweets from February 2015, so findings may not generalise to other periods or countries.
- The dataset covers about eight days of US tweets from 2015, so it is a limited time slice.
- The labels are human judgments, and neutral sentiment is generally harder to agree on.
- 62.7% of tweets are negative, so macro F1 is a fairer metric than accuracy for model comparison.
- Neutral is the hardest class to classify consistently.
- Virgin America has a small sample (504 tweets in total; 181 negative), so its sentiment percentages should be treated cautiously.
- Tweet counts per airline reflect how many tweets were collected, not airline size or customer base.
- People mostly tweet when they are unhappy, which means social media sentiment is not a complete picture of overall customer experience.
- The dataset contains 155 repeated tweet IDs; there were 62 exact copies and 18 IDs with conflicting labels. All 14,640 rows were kept in the analysis, and duplicate removal was used only as a sensitivity check because metrics stayed unchanged.

## Conclusion

This project shows how social-media sentiment analysis can be used to surface the main operational issues customers discuss online. The strongest result is that RoBERTa substantially outperforms VADER on the full dataset, but the real business value is in the complaint themes: customer service, late flights, and cancelled flights dominate the negative sentiment discussion. The analysis is most useful as an indicator of where customers are most dissatisfied and where operational teams should focus their attention next.
