from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

def load():
    if not os.path.exists('data.npy'):
        with open('news.txt', encoding='utf-8') as f:
            news = f.readlines()
        vectorizer = CountVectorizer(stop_words='english', min_df=6, max_df=0.95)
        data = vectorizer.fit_transform(news)
        np.save('data.npy', data.toarray())
        with open('words.txt', 'w', encoding='utf-8') as f:
            f.write(' '.join(vectorizer.get_feature_names()))
    data = np.load('data.npy')
    with open('words.txt', encoding='utf-8') as f:
        words = f.readline().split()
    words = np.array(words)
    return data, words

if __name__ == "__main__":
    data, words = load()
    