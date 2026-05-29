import os
import requests
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    csv_path = os.path.join(BASE_DIR, 'reviews.csv')
    
    if not os.path.exists(csv_path):
        url = "https://itunes.apple.com/us/rss/customerreviews/id=544007664/sortBy=mostRecent/json"
        response = requests.get(url)
        data = response.json()
        
        reviews = []
        for entry in data.get('feed', {}).get('entry', []):
            if 'content' in entry and 'label' in entry['content']:
                text = entry['content']['label'].strip()
                if text:
                    reviews.append(text)
                    
        df = pd.DataFrame({'Review': reviews})
        df.to_csv(csv_path, index=False)
        
    return pd.read_csv(csv_path)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(str(text).lower())
    processed_tokens = []
    
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            lemmatized = lemmatizer.lemmatize(word)
            stemmed = stemmer.stem(lemmatized)
            processed_tokens.append(stemmed)
            
    return " ".join(processed_tokens)

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def plot_sentiment(df):
    counts = df['Sentiment'].value_counts()
    
    plt.figure(figsize=(8, 6))
    colors = ['#4CAF50' if x == 'Positive' else '#F44336' if x == 'Negative' else '#9E9E9E' for x in counts.index]
    bars = plt.bar(counts.index, counts.values, color=colors)
    
    plt.title('Sentiment Distribution of Reviews')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'sentiment_chart.png'))
    plt.show()

if __name__ == "__main__":
    df = load_data()
    
    df['Processed_Review'] = df['Review'].apply(preprocess_text)
    df.to_json(os.path.join(BASE_DIR, 'processed_reviews.json'), orient='records', indent=4, force_ascii=False)
    
    df['Sentiment'] = df['Review'].apply(get_sentiment)
    print(df['Sentiment'].value_counts())
    
    plot_sentiment(df)