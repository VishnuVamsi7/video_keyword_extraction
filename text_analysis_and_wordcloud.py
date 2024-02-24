from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read text from a file
file_path = 'output_audio.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text
tokens = word_tokenize(text.lower())  # Convert to lowercase for consistency

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# Calculate word frequencies
freq_dist = FreqDist(filtered_tokens)

# Display the most common keywords
num_keywords = 500
keywords = freq_dist.most_common(num_keywords)

# Save keywords to a file
output_file_path = '/content/keywords.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for keyword, frequency in keywords:
        output_file.write(f"{keyword}\n")
print(f"Keywords saved to {output_file_path}")

# Convert the list of tuples to a dictionary for WordCloud
wordcloud_dict = dict(keywords)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)

# Plotting the word cloud
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Keywords Word Cloud')
plt.axis('off')
plt.show()
