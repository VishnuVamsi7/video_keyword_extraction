import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Function to read keywords from a file
def read_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        keywords = file.read().splitlines()
    return keywords

# Read text from a file
file_path = 'output_audio.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization with spaCy
doc = nlp(text)

# Stop word removal and lemmatization with NLTK
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

keywords1 = []
for token in doc:
    if token.text.lower() not in stop_words:
        lemma = lemmatizer.lemmatize(token.text)
        keywords1.append(lemma)

keywords1 = list(set(keywords1))

# Save keywords to a text file
output_file_path1 = '/content/keywords1.txt'
with open(output_file_path1, 'w', encoding='utf-8') as output_file:
    output_file.write("\n".join(keywords1))

print(f"Keywords saved to {output_file_path1}")
print(keywords1)

# Tokenize the text
tokens = word_tokenize(text.lower())  # Convert to lowercase for consistency

# Remove stopwords
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# Calculate word frequencies
freq_dist = FreqDist(filtered_tokens)

# Display the most common keywords
num_keywords = 500
keywords2 = [keyword for keyword, _ in freq_dist.most_common(num_keywords)]

# Save keywords to a file
output_file_path2 = '/content/keywords2.txt'
with open(output_file_path2, 'w', encoding='utf-8') as output_file:
    for keyword in keywords2:
        output_file.write(f"{keyword}\n")

print(f"Keywords saved to {output_file_path2}")
print(keywords2)

# Jaccard Similarity
jaccard_similarity = jaccard_score(set(keywords1), set(keywords2))
print(f"Jaccard Similarity: {jaccard_similarity}")

# Cosine Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([" ".join(keywords1), " ".join(keywords2)])
cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
print(f"Cosine Similarity: {cosine_sim}")
