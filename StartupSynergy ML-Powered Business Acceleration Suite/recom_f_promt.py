import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

word2vec_model_path = "GoogleNews-vectors-negative300.bin"  # Path to the Word2Vec model file
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

df = pd.read_csv("buisness_data.csv")

def recommend_categ_for_prompt(prompt):
    prompt_tokens = prompt.lower().split()

    # Calculate average vector representation of the prompt
    prompt_vector = np.mean([word2vec_model[token] for token in prompt_tokens if token in word2vec_model], axis=0)

    # Calculate similarity between the prompt and each business category
    similarities = {}
    for category in df['Category']:
        category_tokens = category.lower().split()
        category_vector = np.mean([word2vec_model[token] for token in category_tokens if token in word2vec_model], axis=0)
        if category_vector is not None and not np.isnan(category_vector).any():
            similarities[category] = cosine_similarity([prompt_vector], [category_vector])[0][0]

    # Get the most similar business category
    if similarities:  # Check if similarities dictionary is not empty
        most_similar_category = max(similarities, key=similarities.get)
        recommended_investors = df[df['Category'] == most_similar_category]['Investors'].values[0]
        return [most_similar_category, recommended_investors]
    else:
        return ["No similar category found", "No recommended investors"]
