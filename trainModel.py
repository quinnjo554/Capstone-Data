from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from gensim.models import KeyedVectors

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize text
    word_tokens = word_tokenize(text)
    # Lemmatize and remove stopwords
    filtered_text = [lemmatizer.lemmatize(word) for word in word_tokens if not word in stop_words]
    return ' '.join(filtered_text)

def compute_wmd(document1, document2, word_embeddings_model):
    # Tokenize the preprocessed text into individual words
    tokens1 = document1.split()
    tokens2 = document2.split()
    
    # Compute the Word Mover's Distance using the word embeddings
    try:
        wmd_distance = word_embeddings_model.wmdistance(tokens1, tokens2)
        return wmd_distance
    except Exception as e:
        print(f"Error computing WMD: {e}")
        return None

# Load the pre-trained GloVe model
word_embeddings_model = api.load("glove-wiki-gigaword-300")


resume = """
John Doe
Fargo, ND | johndoe@gmail.com | 555-555-5555 | LinkedIn: johndoe

Objective: Skilled software engineer seeking a challenging position where I can utilize my experience in Python, Java, and machine learning to develop innovative solutions and contribute to the team.

Skills: Python, Java, Machine Learning, Problem-Solving, Team Collaboration

Work Experience:
Software Engineer | Tech Company | Fargo, ND | June 2016 - Present
- Developed and maintained machine learning models that improved system efficiency by 30%.
- Collaborated with a team of engineers to deliver projects on time and under budget.

Education:
Bachelor's degree in Computer Science | University Name | May 2016

Certifications:
Certified Python Developer | PCEP | 2020
Java SE 8 Programmer Certification | Oracle | 2021
"""

jd = """
Job Title: Software Engineer

Job Summary: We are looking for a skilled software engineer to join our innovative tech team. The ideal candidate has experience in Python, Java, and machine learning.

Responsibilities:
- Develop and maintain machine learning models.
- Collaborate with a team of engineers to deliver projects on time and under budget.

Requirements:
- Bachelor's degree in Computer Science or related field.
- Experience in Python, Java, and machine learning.
- Strong problem-solving skills.

Benefits:
- Competitive salary.
- Health, dental, and vision insurance.
- Generous vacation policy.
""" 

resume = preprocess_text(resume)
jd = preprocess_text(jd)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
vectorizer.fit([resume, jd])

resume_tfidf = vectorizer.transform([resume])
jd_tfidf = vectorizer.transform([jd])

similarity_score_tfidf = cosine_similarity(resume_tfidf, jd_tfidf)[0][0]
print(f"Similarity Score (TF-IDF): {similarity_score_tfidf}")

wmd_score = compute_wmd(resume, jd, word_embeddings_model)
if wmd_score is not None:
    print(f"Word Mover's Distance: {wmd_score}")
else:
    print("Error computing Word Mover's Distance.")

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define your fine-tuning task
# For example, let's say we want to fine-tune BERT for sentiment analysis

# Prepare your data
train_texts = ['Example text 1', 'Example text 2', ...]
train_labels = [0, 1, ...]  # 0 for negative sentiment, 1 for positive sentiment

train_encodings = tokenizer(train_texts, truncation=True, padding=True)

train_labels = torch.tensor(train_labels)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(**train_encodings, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Evaluate the fine-tuned model
# For example, using a validation set
val_texts = ['Validation text 1', 'Validation text 2', ...]
val_labels = [0, 1, ...]  # 0 for negative sentiment, 1 for positive sentiment

val_encodings = tokenizer(val_texts, truncation=True, padding=True)
val_labels = torch.tensor(val_labels)

model.eval()
with torch.no_grad():
    outputs = model(**val_encodings)
    predictions = torch.argmax(outputs.logits, dim=1)
    accuracy = (predictions == val_labels).float().mean()

print(f"Validation accuracy: {accuracy}")
 