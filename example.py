import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- Create dummy objects for the metadata ---

# Example: a list of product IDs
product_ids = ["P1", "P2", "P3", "P4", "P5"]

# Create a LabelEncoder and fit it on product IDs.
product_encoder = LabelEncoder()
product_encoder.fit(product_ids)

# Build a reverse mapping: index to product ID.
idx_to_product = {idx: pid for idx, pid in enumerate(product_encoder.classes_)}

# Create a simple mapping from product ID to product name.
product_to_name = {
    "P1": "Ultra HD Television",
    "P2": "Wireless Headphones",
    "P3": "Coffee Maker",
    "P4": "Smartphone",
    "P5": "Electric Kettle"
}

# Create a dummy rating scaler based on rating values 1 through 5.
ratings = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
rating_scaler = MinMaxScaler().fit(ratings)

# Create a dummy category mapping:
# For simplicity, we map the encoded indices to categories.
# For example: indices 0 and 1 belong to "Electronics", 2 to "Home", etc.
category_mapping = {
    0: "Electronics",
    1: "Electronics",
    2: "Home",
    3: "Electronics",
    4: "Kitchen"
}

# Create dummy embeddings.
# Let's assume an embedding dimension of 16.
product_embeddings = np.random.rand(len(product_ids), 16)
# For the user embeddings, suppose we have 10 dummy users.
user_embeddings = np.random.rand(10, 16)

# --- Bundle metadata into a dictionary ---

metadata = {
    "product_encoder": product_encoder,
    "idx_to_product": idx_to_product,
    "product_to_name": product_to_name,
    "rating_scaler": rating_scaler,
    "category_mapping": category_mapping,
    "user_embeddings": user_embeddings,
    "product_embeddings": product_embeddings
}

# --- Save the metadata to a pickle file ---
with open("amazon_recommender_metadata_updated.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Metadata saved successfully.")
