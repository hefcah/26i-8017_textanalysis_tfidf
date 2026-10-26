"""
Unstructured Text Analysis & Automated Triage with TF-IDF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load 4 real-world unstructured user discussion categories
CATEGORIES = [
    "sci.electronics",
    "rec.autos",
    "comp.sys.mac.hardware",
    "sci.med",
]

print("Loading dataset...")
raw_data = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
)

df = pd.DataFrame(
    {
        "text": raw_data.data,
        "true_label": [raw_data.target_names[i] for i in raw_data.target],
    }
)

# retention of posts with meaningful length (> 50 characters)
df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)
print(f"Loaded {len(df)} unstructured user posts.\n")


# ## Step 1: Text Cleaning, Tokenization & Normalization
sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

# TODO 1.1: Tokenize, lowercase, and keep alphabetic tokens
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2: Remove standard English stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: Apply PorterStemmer and WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens]

if filtered_tokens is not None:
    print("Filtered Tokens:  ", filtered_tokens[:12])
    print("Stemmed Roots:    ", stemmed_tokens[:12])
    print("Lemmatized Words: ", lemmatized_tokens[:12])


# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction
bow_vec = CountVectorizer(stop_words='english', max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words='english', max_features=4000, min_df=5)

X_bow = bow_vec.fit_transform(df['text'])
X_tfidf = tfidf_vec.fit_transform(df['text'])

# Sparsity variable required by test suite
total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")


# ## Step 3: Automated Topic Discovery & Spatial Mapping
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_tfidf)


# ## Step 4: Problem-Solving Diagnostics & Explainability
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    Return the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id`. Returns a list of (term, weight) tuples.
    """
    row = matrix.getrow(doc_id).toarray().ravel()
    top_indices = row.argsort()[-top_n:][::-1]
    feature_names = vectorizer.get_feature_names_out()
    
    explanation = [(feature_names[idx], float(row[idx])) for idx in top_indices if row[idx] > 0]
    return explanation