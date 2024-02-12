from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

def generate_document_embedding(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token in word_embeddings_model and token not in stop_words]
    return sum(word_embeddings_model[token] for token in tokens) / len(tokens) if tokens else None



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
The ideal candidate for this position is a self-driven individual that likes to make a big impact in
small teams - and has a passion for both software development and aviation. The opportunity
will primarily involve a mix of frontend development in React and API development in .NET for
Appareo's FOQA solution, EnVision. This individual will coordinate with a cross-disciplinary
software team to help make solutions that work well from embedded software to the cloud.
Essential Duties and Responsibilities
The essential functions include, but are not limited to the following:
Implement new software features in an iterative process
Help maintain and continuously update EnVision website infrastructure and ecosystem
Assist with design and estimation of new features, identifying areas of risk
Work requires occasional travel to meetings, site visits, and conferences
Minimum Qualifications (Knowledge, Skills, and Abilities)
B.S. in Computer Science and/or training or equivalent combination of education and experience
General professional programming, 5+ years experience
Strong proficiency in React, 2+ years experience
Proficiency in C# and .NET, 2+ years experience
Experience with relational database design
Experience with Docker
Bonus: Aviation domain experience (e.g. private pilot etc.)
Bonus: Ansible experience
Bonus: AWS services experience
Bonus: enjoys algorithms
Why Work at Appareo Systems?
Appareo Systems is focused on dramatic growth and innovation. With that in mind, Appareo Systems is committed to providing opportunities for individual growth and career satisfaction and assisting employees to realize their potential by providing appropriate training and development opportunities.
Culture is Everything
Our culture is deeply rooted in our company purpose and core values. We love what we do and we’re passionate about it. Everyone at Appareo is helping to build a company that is meaningful and impactful. We are defining the direction of a rapidly growing business. We work hard, but we also have a little fun along the way.
""" 
##get keyt words simularity
resume = preprocess_text(resume)
jd = preprocess_text(jd)

resume_embedding_bert = generate_document_embedding_bert(resume)
jd_embedding_bert = generate_document_embedding_bert(jd)
resume_embedding = generate_document_embedding(resume)
jd_embedding = generate_document_embedding(jd)

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
