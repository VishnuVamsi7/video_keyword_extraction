import cv2
import easyocr
import spacy
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Code 1: OCR on video frames
video_path = '/content/drive/MyDrive/Colab Notebooks/SP_Project/Videos/Untitled video - Made with Clipchamp.mp4'
cap = cv2.VideoCapture(video_path)
reader = easyocr.Reader(['en'], gpu=True)
output_file_path = 'output_text.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = reader.readtext(frame)
        for (bbox, text, prob) in results:
            frame_text = f"Frame {frame_number}: Text: {text}, Probability: {prob:.2f}"
            print(frame_text)
            output_file.write(text + '\n')
        frame_number += 1

cap.release()
print(f"Extracted text saved to {output_file_path}")

# Code 2: NLP processing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 7000000
stop_words = set(stopwords.words("english"))

with open('output_text.txt', 'r') as file:
    text = file.read()

doc = nlp(text)
keywords1 = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
keywords1 = list(set(keywords1))
output_file_path1 = 'output_text_keywords.txt'

with open(output_file_path1, 'w', encoding='utf-8') as output_file:
    for keyword in keywords1:
        output_file.write(keyword + '\n')

print(f"Keywords saved to {output_file_path1}")

# Code 4: Keyword frequency analysis
tokens = word_tokenize(text.lower())
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
freq_dist = FreqDist(filtered_tokens)
num_keywords = 500
keywords2 = [keyword for keyword, _ in freq_dist.most_common(num_keywords)]
output_file_path2 = '/content/keywords3.txt'

with open(output_file_path2, 'w', encoding='utf-8') as output_file:
    for keyword, frequency in freq_dist.most_common(num_keywords):
        output_file.write(f"{keyword}\n")

print(f"Keywords saved to {output_file_path2}")

# Code 5: Word cloud and bar graph visualization
wordcloud_dict = dict(freq_dist.most_common(num_keywords))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)

plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Keywords Word Cloud')
plt.axis('off')
plt.show()

words, frequencies = zip(*freq_dist.most_common(num_keywords))
plt.figure(figsize=(10, 6))
plt.barh(words[:30], frequencies[:30], color='skyblue')
plt.title('Top Keywords Bar Graph')
plt.xlabel('Frequency')
plt.ylabel('Keywords')
plt.gca().invert_yaxis()
plt.show()

print(f"Keywords and visualizations saved to {output_file_path2}")
