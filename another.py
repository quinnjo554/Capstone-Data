from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = en_core_web_sm

def getResumeMatchScore(job, resume, positionText, resumeWords, positionWords):
    '''
    Key Word Similarity Calculation
    Iterate through the most common words in your resume 
    and calculate the similarity scores for each word 
    in relation to the most common words in the job description
    Find the max similarity score of each word and calculate
    the average of the scores in the end
    '''
    print(job[1], job[0])
    print(job[-1])
    print()
    print(positionWords)

    overallSimilarity = 0
    maxSims = []

    for i, word in enumerate(resumeWords):
        sims = []

        for j, word2 in enumerate(positionWords):
            sim = nlp(resumeWords[i][0]).similarity(nlp(positionWords[j][0]))
            if sim == 1: sim = sim * 1.5
            sims.append(sim)

        maxSims.append(max(sims))

    keyWordSimilarity = round(sum(maxSims) / len(maxSims) * 100, 2)
    overallSimilarity = keyWordSimilarity
    print('\nKey Word Similarity:', keyWordSimilarity)

    '''
    Sentence Similarity Calculation
    Iterate through each line of your resume and find the sentence 
    that is most similar to it in the job description 
    Find the similarity values of the 15 most similar sentences and
    calculate the average of those values 
    '''
    maxSentSims = []

    for line in resume:
        if len(line) >= 30:
            sentSims = []
            sents = []

            for sent in positionText:
                if len(sent) >= 10:
                    s = sentenceSimilarity(line, sent)
                    sentSims.append(s)
                    sents.append(line + ' ' + sent)

            maxSentSims.append(max(sentSims))

    maxSentSims.sort(reverse=True)
    sentSimilarity = round(sum(maxSentSims[0:15]) / len(maxSentSims[0:15]) * 100, 2)
    overallSimilarity += sentSimilarity 

    print('Sentence Similarity:', sentSimilarity)

    print('\nOverall Score:', overallSimilarity)
    print()

    # Convert resume and position text to a list of strings
    resume_text = ' '.join(resume)
    position_text = ' '.join(positionText)

    # Use TF-IDF Vectorizer to calculate similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, position_text])
    similarity_score_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    print(f"Similarity Score (TF-IDF): {similarity_score_tfidf}")

    # Calculate similarity using BERT embeddings
    similarity_score_bert = calculate_similarity_bert(resume_text, position_text)
    print(f"Similarity Score (BERT): {similarity_score_bert}")

    return keyWordSimilarity, sentSimilarity, overallSimilarity, similarity_score_tfidf, similarity_score_bert

resume = """
"""

jd = """

"""

getResumeMatchScore(job, resume, jd, resumeWords, positionWords)