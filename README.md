Job Description and Resume Similarity using Word Embeddings
This Python script calculates the similarity between a job description (JD) and a resume using Word Embeddings. It preprocesses the text data, generates document embeddings, and computes the cosine similarity between them.

Requirements
Python 3.x
NLTK (Natural Language Toolkit)
Gensim
scikit-learn
Install the required Python packages using pip:

bash
Copy code
pip install nltk gensim scikit-learn
Usage
Make sure you have NLTK's stopwords corpus and WordNet lemmatizer downloaded by running the following commands:

bash
Copy code
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
Prepare your resumes and job descriptions in plain text format and update the resume and jd variables in the script with your text data.

Run the script:

bash
Copy code
python similarity.py
The script will output the similarity score between the resume and the job description using Word Embeddings.

Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
