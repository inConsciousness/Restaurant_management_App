import pandas as pd
import numpy as np
#1 datacollection and preprocessing
from sklearn.model_selection import train_test_split
customers = pd.read_csv('/customers.csv')
orders = pd.read_csv('/orders.csv')
menu = pd.read_csv('/menu.csv')
reviews = pd.read_csv('/reviews.csv')
customers=pd.get_dummies(customers)
menu=pd.get_dummies(menu)
data = pd.merge(orders, customers, on='customer_id')
data = pd.merge(data, menu, on='item_id')
data = pd.merge(data, reviews, on=['customer_id', 'item_id'])
train_data,test_data=train_test_split(data,test_size=0.2,random_state=42)
#2 nlp_for_text_analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

tfidf_vectorizer = TfidfVectorizer(max_features=500)
tfidf_reviews = tfidf_vectorizer.fit_transform(reviews['review_text']).toarray()
tfidf_descriptions = tfidf_vectorizer.fit_transform(menu['description']).toarray()

# Add text features to the dataset
reviews_tfidf = pd.DataFrame(tfidf_reviews, columns=tfidf_vectorizer.get_feature_names_out())
descriptions_tfidf = pd.DataFrame(tfidf_descriptions, columns=tfidf_vectorizer.get_feature_names_out())

# Combine text features with existing data
train_data = pd.concat([train_data, reviews_tfidf, descriptions_tfidf], axis=1)
#3 model training with tensor flow
import tensorflow as tf
from keras import layers, models

# Define model architecture
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(train_data.shape[1] - 1,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#4

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate menu description
input_text = "Customer prefers vegan dishes with spicy flavors and has positive reviews for similar items."
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

#5

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Load pre-trained RAG model
tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', index_name="custom_index")
model = RagTokenForGeneration.from_pretrained('facebook/rag-token-nq')

# Retrieve relevant documents
input_text = "Suggest a menu item for a vegan customer who likes spicy food and has given positive reviews for similar items."
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']
retrieved_docs = retriever(input_ids)

# Generate enhanced menu suggestion
output = model.generate(input_ids, context_input_ids=retrieved_docs, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

#6

def recommend_menu(customer_profile, reviews):
    # Predict preferences using the trained deep learning model
    customer_vector = pd.get_dummies(customer_profile)
    preference_score = model.predict(customer_vector)

    # Generate menu suggestion using the generative model
    input_text = f"Customer profile: {customer_profile}. Reviews: {reviews}. Preference score: {preference_score}."
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Enhance recommendation with RAG
    retrieved_docs = retriever(input_ids)
    output = model.generate(input_ids, context_input_ids=retrieved_docs, max_length=50)
    enhanced_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return enhanced_text

# Test the recommendation system
customer_profile = {"age": 30, "gender": "female", "preference": "vegan", "likes_spicy": True}
reviews = "The customer likes vegan dishes and has given positive reviews for spicy food."
recommendation = recommend_menu(customer_profile, reviews)
print(recommendation)
