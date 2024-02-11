from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    #tokenize text
    word_tokens = word_tokenize(text)
    #lemmatize and remove stopwords
    filtered_text = [lemmatizer.lemmatize(word) for word in word_tokens if not word in stop_words]
    return ' '.join(filtered_text)

resume = ""
jd = ""

resume = preprocess_text(resume)
jd = preprocess_text(jd)

word_embeddings_model = api.load("glove-wiki-gigaword-300")

def generate_document_embedding(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token in word_embeddings_model and token not in stop_words]
    #calculate average word embeddings
    if tokens:
        return sum(word_embeddings_model[token] for token in tokens) / len(tokens)
    else:
        return None

resume_embedding = generate_document_embedding(resume)
jd_embedding = generate_document_embedding(jd)

if resume_embedding is not None and jd_embedding is not None:
    #calculate cosine similarity between the embeddings
    similarity_score_word_embeddings = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    print(f"Similarity Score (Word Embeddings): {similarity_score_word_embeddings}")
else:
    print("One or both document embeddings could not be generated.")
