from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    word_tokens = word_tokenize(text)
    filtered_text = [lemmatizer.lemmatize(word) for word in word_tokens if not word in stop_words]
    return ' '.join(filtered_text)

word_embeddings_model = api.load("glove-wiki-gigaword-300")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def generate_document_embedding_bert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach()

def generate_document_embedding(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token in word_embeddings_model and token not in stop_words]
    return sum(word_embeddings_model[token] for token in tokens) / len(tokens) if tokens else None

def get_most_common_words(text, n=10):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    freq_dist = nltk.FreqDist(tokens)
    return [word for word, freq in freq_dist.most_common(n)]

def calculate_similarity_bert(text1, text2):
    text1_common_words = get_most_common_words(text1)
    text2_common_words = get_most_common_words(text2)
    text1_common_str = ' '.join(text1_common_words)
    text2_common_str = ' '.join(text2_common_words)
    text1_embedding = get_bert_embeddings(text1_common_str)
    text2_embedding = get_bert_embeddings(text2_common_str)
    text1_embedding_tensor = torch.tensor(text1_embedding).clone().detach()
    text2_embedding_tensor = torch.tensor(text2_embedding).clone().detach()
    return cosine_similarity(text1_embedding_tensor.unsqueeze(0), text2_embedding_tensor.unsqueeze(0))[0][0]


resume = """ """

jd = """
2 to 12 Years,BCA,$56K-$116K,Ashgabat,Turkmenistan,38.9697,59.5563,Intern,100340,2022-12-19,Female,Francisco Larsen,461-509-4216,Web Developer,Frontend Web Developer,Idealist,"Frontend Web Developers design and implement user interfaces for websites, ensuring they are visually appealing and user-friendly. They collaborate with designers and backend developers to create seamless web experiences for users.","{'Health Insurance, Retirement Plans, Paid Time Off (PTO), Flexible Work Arrangements, Employee Assistance Programs (EAP)'}","HTML, CSS, JavaScript Frontend frameworks (e.g., React, Angular) User experience (UX)","Design and code user interfaces for websites, ensuring a seamless and visually appealing user experience. Collaborate with UX designers to optimize user journeys. Ensure cross-browser compatibility and responsive design.",PNC Financial Services Group,"{""Sector"":""Financial Services"",""Industry"":""Commercial Banks"",""City"":""Pittsburgh"",""State"":""Pennsylvania"",""Zip"":""15222"",""Website"":""www.pnc.com"",""Ticker"":""PNC"",""CEO"":""William S. Demchak""}"
""" 


##get key words similarity use a hashmap <string, int> and ++ to the key when u find a word in the map
resume = preprocess_text(resume)
jd = preprocess_text(jd)

resume_embedding_bert = generate_document_embedding_bert(resume)
jd_embedding_bert = generate_document_embedding_bert(jd)
resume_embedding = generate_document_embedding(resume)
jd_embedding = generate_document_embedding(jd)
similarity_score_bert = 0
similarity_score_word_embeddings = 0
if resume_embedding_bert is not None and jd_embedding_bert is not None:
    similarity_score_bert = cosine_similarity([resume_embedding_bert], [jd_embedding_bert])[0][0]
    print(f"Similarity Score (BERT): {similarity_score_bert}")
else:
    print("One or both BERT document embeddings could not be generated.")

if resume_embedding is not None and jd_embedding is not None:
    similarity_score_word_embeddings = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    print(f"Similarity Score (Word Embeddings): {similarity_score_word_embeddings}")
else:
    print("One or both Word Embeddings document embeddings could not be generated.")

similarity = calculate_similarity_bert(resume, jd)
print(f"Key Word Similarity: {similarity}")

keyword_weight = 0.6
tfidf_weight = 0.3
embedding_weight = 0.1 

combined_score = (keyword_weight * similarity) + (tfidf_weight * similarity_score_bert) + (embedding_weight * similarity_score_word_embeddings)

print("Combined Score:", combined_score)

## This is my starter code.
## Calculates cosign sim
## Improvements or things to try
## Try Stemming instead of lemma partitioning
## Keep verbs and nouns but remove unrelated parts of speech
## Handle named entities. Skills, educations, companies ect. Fuzzy matching?
## TF-ID instead of simple word count. *****important****
## Word2Vec or GloVe? maybe even Gensim
## Experiment with sentence-level embeddings
##
##
## jaccard similarity?
##
##