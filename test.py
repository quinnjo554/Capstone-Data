from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from gensim.models import KeyedVectors

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
Brian Jacob
Westfield, MA
Senior Frontend Engineer with over 12 years of experience in web engineering and e-commerce development. Skilled in creating high-performing and visually appealing user interfaces for Shopify storefronts using React, Vue, Shopify, Liquid, HTML, CSS, and JavaScript. Proficient in custom theme development, responsive design, performance optimization, and cross-browser compatibility. A collaborative team player with a passion for delivering exceptional user experiences and driving business growth. Strong problem-solving abilities and a proven track record of successfully delivering projects on time and exceeding client expectations.
Work Experience
Senior Frontend Engineer
Tata Consultancy Services (TCS)
April 2021 to Present
• Orchestrated the development of the ERP system that delivered high-quality releases with 98%+ uptime and 0 critical bugs using Apollo GraphQL, React, Tailwind CSS and TypeScript.
• Led the successful migration of a React frontend application to Next.js 13, leveraging the power of app routing to enhance performance, enable server-side rendering (SSR), and optimize search engine optimization (SEO).
• Took full ownership of the customer admin portal and IPP part of the application built using React, Next.js, TypeScript, MUI and Styled Component.
• Implemented micro-frontend pattern and managed collection of sample components using Storybook.
• Created unit and integration test cases using Jest, Selenium and react testing library, and automated acceptance tests using Jenkins for streamlined deployments.
• Built a financial auditing system for Stripe payments using React, Laravel, and PHP where the client payment reconciliation was performed between the application database and Stripe's API.
• Maintained ecommerce website for a medium-sized company, ensuring that all products were displayed correctly and the site was updated regularly with new content using Shopify and React.
• Implemented Headless architecture to customize ecommerce platform to meet the specific needs of the company, including integrating customer loyalty programs and third-party shipping software.
• Optimized ecommerce platform for better search engine visibility and faster page loading times; saw a 10% increase in traffic within 2 months after implementation of optimizations.
• Successfully led the migration of “supermouth.com” from a traditional Shopify setup to a headless Hydrogen Shopify architecture, incorporating React, Remix, TypeScript, and Tailwind CSS, resulting in improved performance and enhanced development capabilities.
• Maintained and updated the "Software Testing and QA Service" based on feedback from customers using Node.js for its backend and React 16 for the frontend.
• Collaborated with UI/UX designers to create visually appealing user interfaces and optimized user experience, resulting in a 15% increase in user satisfaction.

Full Stack Developer
ASM International
March 2017 to April 2021
• Built and maintained an online adhesive analyzing platform “ARL Adhesives” using Next.js, React, Node.js, Redux, S3, TypeScript and PostgreSQL and worked on multiple new features, including analyzing materials, data visualization and market segmentation.
• Maintained and released a microservices architecture using Redis and Docker on an Ubuntu platform using HTTP/REST interfaces with deployment into a multi-node Kubernetes environment and decreased monthly traffic by 30%.
• Utilized and maintained fully automated CICD pipelines for code deployment using CircleCI for 99.9% uptime.
• Implemented automation scripts to add translation helpers to raw strings in codebase and added new unit and integration tests to confirm that i18n and l10n work as expected.
• Participated in peer code review for PR from other developers on a daily basis.
• Developed and tested many features in an agile environment using Cypress, RSpec.
• Managed and executed on the product strategy, master both the Shopify and liquid language to execute a cohesively coded and well-designed webpage.
• Developed a high performance customer facing ecommerce application using React, SASS, Laravel and MongoDB, resulting in a potential increase of customer satisfaction by 40%.
• Redesigned the application modules using Vue, Python, Django, Rest API/Services which resulted in 30% decrease in response time, 25% less code and 26% increase in revenue.

Shopify and React Developer
Boston Consulting Group (BCG)
June 2015 to March 2017
• Worked on the payments team to save over 50,000 customers time and improve cash flow through the development of modern, responsive customer experiences using React, Redux, Node.js and the Material UI Library.        
• Used Mocha, React Testing Library and Cypress to create test driven development (TDD).
• Expanded features, refined code, and improved performance with React, producing smoother operations and enhancing user engagement.
• Created ecommerce sites integrated with PayPal, Authorize.net, and other payment APIs using the Shopify, React and Amazon DynamoDB.
• Managed code versioning with GitHub and deployment to staging and production servers.
• Introduced wider use of isomorphic React, Node.js and MongoDB for web applications, decreasing load times by roughly 35%.
• Involved in all phases of the Software development life cycle (SDLC) using Agile Methodology.
• Maintained backend code built with Node.js, ExpressJS, and MySQL also collaborates with the product team to implement new features planned for future products.
• Refactored error message handling design by implementing React alert dialog resulting in a potential decrease in user input errors by 40%.

Frontend Developer
Boston Consulting Group (BCG)
October 2012 to May 2015
• Contributed to the in-house UI library to create reusable React components that saved over 200 hours of development time per month.
• Worked with an agile team to migrate the legacy company website to AngularJS, Sass, and Drupal.
• Created a web app MVP for a store delivery management platform with business customers to create, manage, and monitor deliveries using React.
• Planned and engineered RESTful web services to manipulate dynamic datasets using Node.js and Express Revamped main pages of 3+ businesses and agencies using React and JQuery.
• Collaborated with the UX/UI design teams to improve the website and applications, increasing conversion rates by 25% using SASS/SCSS and Bootstrap.

Education
Bachelor's degree of Computer Science in Computer Science
Smith College - Northampton, MA
2008 to 2012
Skills
•       Selenium
•       PostgreSQL
•       Problem Solving
•       WordPress
•       HTML
•       Redis
•       Kanban
•       Git
•       Active listening
•       Next
•       Teamwork and Collaboration
•       Emotional intelligence
•       Quick learning
•       Cypress
•       Dedication
•       Creativity
•       ElasticSearch
•       Laravel
•       Attention to Detail
•       Microsoft Azure
•       Kubernetes
•       Agile
•       TypeScript
•       Scrum
•       Vue
•       Good Communication
•       Leadership
•       React
•       MySQL
•       Celery
•       Jira
•       JavaScript
•       GCP
•       CSS
•       Shopify
•       AWS
•       JQuery
•       Magento
•       Flexibility
•       PHP
•       Docker
•       CircleCI, GitHub Actions
•       Time management
•       CakePHP
•       RabbitMQ
•       Apache Kafka
•       MongoDB, Amazon DynamoDB
•       APIs
•       C/C++
•       Test Cases
•       SQL
•       Ant
•       Apache
•       Microsoft SQL Server
•       Web Services
•       Bootstrap
•       Angular
•       Java
•       Software Development
•       User Interface (UI)
•       CI/CD
•       JSON
•       Jenkins
•       Node.js
•       REST
•       XML
"""

jd = """
2 to 12 Years,BCA,$56K-$116K,Ashgabat,Turkmenistan,38.9697,59.5563,Intern,100340,2022-12-19,Female,Francisco Larsen,461-509-4216,Web Developer,Frontend Web Developer,Idealist,"Frontend Web Developers design and implement user interfaces for websites, ensuring they are visually appealing and user-friendly. They collaborate with designers and backend developers to create seamless web experiences for users.","{'Health Insurance, Retirement Plans, Paid Time Off (PTO), Flexible Work Arrangements, Employee Assistance Programs (EAP)'}","HTML, CSS, JavaScript Frontend frameworks (e.g., React, Angular) User experience (UX)","Design and code user interfaces for websites, ensuring a seamless and visually appealing user experience. Collaborate with UX designers to optimize user journeys. Ensure cross-browser compatibility and responsive design.",PNC Financial Services Group,"{""Sector"":""Financial Services"",""Industry"":""Commercial Banks"",""City"":""Pittsburgh"",""State"":""Pennsylvania"",""Zip"":""15222"",""Website"":""www.pnc.com"",""Ticker"":""PNC"",""CEO"":""William S. Demchak""}"
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

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def generate_document_embedding_bert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()

resume_embedding_bert = generate_document_embedding_bert(resume)
jd_embedding_bert = generate_document_embedding_bert(jd)

if resume_embedding_bert is not None and jd_embedding_bert is not None:
    similarity_score_bert = cosine_similarity(resume_embedding_bert, jd_embedding_bert)[0][0]
    print(f"Similarity Score (BERT): {similarity_score_bert}")
else:
    print("One or both document embeddings could not be generated.")

