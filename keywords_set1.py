# Read the content of the uploaded text file
with open('output_audio.txt', 'r') as file:
    text = file.read()
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Read text from a file
file_path = 'output_audio.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization with spaCy
doc = nlp(text)

# Stop word removal and lemmatization with NLTK
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

keywords = []
for token in doc:
    if token.text.lower() not in stop_words:
        lemma = lemmatizer.lemmatize(token.text)
        keywords.append(lemma)

keywords = list(set(keywords))

# Save keywords to a text file
output_file_path = '/content/keywords1.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write("\n".join(keywords))

print(f"Keywords saved to {output_file_path}")
print(keywords)
