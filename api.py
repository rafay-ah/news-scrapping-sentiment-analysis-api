
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi import FastAPI, Query, Body, Request,HTTPException
import pandas as pd
import requests

api_key = "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"
API_URL = "https://api-inference.huggingface.co/models/hassan4830/xlm-roberta-base-finetuned-urdu"
headers = {"Authorization": f"Bearer {api_key}"}

app = FastAPI()

@app.get("/")
def read_root():
    return {"Health_Check": "API is Working1"}


@app.get("/scraper/{channel}")
async def scrapper(channel:str):
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    process.crawl(channel)
    process.start()
    scrapped_data = pd.read_csv(f"{channel}.csv")
    return scrapped_data


@app.get("/classify/{channel}")
async def perform_sentiment_analysis(channel:str):
    sid_obj = SentimentIntensityAnalyzer()
    sentiments = []
    scrapped_data = pd.read_csv(f"{channel}.csv")
    details = scrapped_data['Details']

    for detail in details:
        sentiment_dict = sid_obj.polarity_scores(detail)
        negative = sentiment_dict['neg']
        neutral = sentiment_dict['neu']
        positive = sentiment_dict['pos']
        compound = sentiment_dict['compound']

        if sentiment_dict['compound'] >= 0.05:
            overall_sentiment = "Positive"

        elif sentiment_dict['compound'] <= - 0.05:
            overall_sentiment = "Negative"

        else:
            overall_sentiment = "Neutral"
        sentiments.append(overall_sentiment)

    return  sentiments


@app.get("/classify-urdu/{channel}")
async def perform_urdu_sentiment_analysis(channel:str):
    sentiments = []
    scrapped_data = pd.read_csv(f"{channel}.csv")
    details = scrapped_data['Details']

    for detail in details:
        output = query({
            "inputs": f"{detail}",
        })
        sentiments.append(output)

    return sentiments


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response = response.json()
    identified_label = str(response[0][0].values())
    if 'LABEL_0' in identified_label :
        return "negative"
    else:
        return "positive"



