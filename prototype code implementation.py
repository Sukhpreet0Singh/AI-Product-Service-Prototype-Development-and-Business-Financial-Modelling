#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[12]:


# Sample vendor dataset
vendors = pd.DataFrame({
    'VendorID': [1, 2, 3, 4],
    'VendorName': ['Elegant Banquet Hall', 'Budget-Friendly Catering', 'Luxury Decor Services', 'Live Band'],
    'Services': [
        'Spacious venue for weddings and corporate events',
        'Affordable catering with custom menus',
        'Premium decorations for high-end events',
        'Live music for weddings and parties'
    ],
    'BudgetRange': ['High', 'Low', 'High', 'Medium']
})

# Mock user input
user_input = {
    'preferences': "I need a spacious venue with luxury decorations and live music",
    'budget': "High"
}


# In[13]:


# Preprocess text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

vendors['ProcessedServices'] = vendors['Services'].apply(preprocess_text)
user_preferences = preprocess_text(user_input['preferences'])


# In[14]:


# Compute similarity using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(vendors['ProcessedServices'].tolist() + [user_preferences])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Add similarity scores to vendor data
vendors['SimilarityScore'] = cosine_sim

# Filter based on budget and sort by similarity
recommended_vendors = vendors[vendors['BudgetRange'] == user_input['budget']].sort_values(by='SimilarityScore', ascending=False)

print("Recommended Vendors:")
print(recommended_vendors[['VendorName', 'SimilarityScore']])


# In[ ]:




