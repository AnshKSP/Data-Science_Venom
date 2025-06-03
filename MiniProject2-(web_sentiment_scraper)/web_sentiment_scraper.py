import wikipedia
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def get_summary(topic, sentences=3):
    try:
        return wikipedia.summary(topic, sentences=sentences)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"[Ambiguous] '{topic}' has multiple meanings: {', '.join(e.options[:3])}"
    except wikipedia.exceptions.PageError:
        return f"[Error] No page found for '{topic}'."
    except Exception as e:
        return f"[Error] Could not retrieve '{topic}': {str(e)}"

def analyze_sentiment(text):
    tb = TextBlob(text).sentiment.polarity
    vd = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    tb_sent = "positive 😀" if tb > 0 else "negative 😡" if tb < 0 else "neutral 😐"
    vd_sent = "positive 😀" if vd > 0.05 else "negative 😡" if vd < -0.05 else "neutral 😐"
    return tb_sent, vd_sent

def main():
    topics = [t.strip() for t in input("Enter topics (comma separated): ").split(",") if t.strip()]
    with open("summaries.txt", "w", encoding="utf-8") as f:
        for topic in topics:
            print(f"Fetching: {topic}")
            summary = get_summary(topic)
            tb, vd = analyze_sentiment(summary)
            f.write(f"### {topic} ###\n{summary}\nTextBlob: {tb} | Vader: {vd}\n\n")
    print("✅ Summaries and sentiment saved to summaries.txt")

if __name__ == "__main__":
    main()
