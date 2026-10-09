import os, html, re, time
import pandas as pd
from transformers import pipeline

def roberta_prep(text):
    text = html.unescape(text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"http\S+", "http", text)
    return text.strip()

df = pd.read_csv("data/Tweets.csv")
texts = [roberta_prep(t) for t in df["text"]]

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)

labels, scores = [], []
start = time.time()
for i in range(0, len(texts), 500):
    out = pipe(texts[i:i+500], batch_size=32, truncation=True, max_length=128)
    labels += [o["label"].lower() for o in out]
    scores += [o["score"] for o in out]
    print(f"{min(i+500, len(texts))}/{len(texts)} done, {time.time()-start:.0f}s", flush=True)

os.makedirs("outputs", exist_ok=True)
pd.DataFrame({"tweet_id": df["tweet_id"], "roberta": labels, "roberta_score": scores}) \
  .to_csv("outputs/roberta_results.csv", index=False)
print("saved")