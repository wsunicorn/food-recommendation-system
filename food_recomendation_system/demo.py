"""Simple demo runner for the recommendation modules."""
from .data_loader import load_data
from .content_based import ContentBasedRecommender
import os


def run_demo():
    base = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base, 'data', 'Dataset_for_print.csv')
    df = load_data(data_path)
    print('Loaded', len(df), 'rows')

    titles = df['title'].astype(str).tolist()
    desc = df['description'].astype(str).tolist()

    rec = ContentBasedRecommender(method='tfidf')
    rec.fit(titles, desc)
    idx, scores = rec.recommend(0, top_k=5)
    print('Recommendations for item 0:')
    for i, s in zip(idx, scores):
        print(i, s, titles[i][:60])


if __name__ == '__main__':
    run_demo()
