"""
Recommendation utility functions for the e-commerce application.
These functions handle the recommendation logic, model interactions, and similarity calculations.
"""

import numpy as np
import tensorflow as tf
import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    """
    A recommendation engine that uses Deep Matrix Factorization to provide personalized
    product recommendations with category-based enhancements.
    """
    
    def __init__(self, model_path='amazon_recommender_model_updated.keras', metadata_path='amazon_recommender_metadata_updated.pkl'):
        """
        Initialize the recommendation engine.
        
        Args:
            model_path (str): Path to the trained model file
            metadata_path (str): Path to the metadata pickle file
        """
        self.model = None
        self.metadata = None
        self.product_embeddings = None
        self.user_embeddings = None
        self.product_encoder = None
        self.idx_to_product = None
        self.product_to_name = None
        self.category_mapping = None
        self.df = None
        self.category_weight = 0.6  # Default weight for category bonus
        
        # Load model and metadata
        self.load_model(model_path)
        self.load_metadata(metadata_path)
    
    def load_model(self, model_path):
        """
        Load the trained recommendation model.
        
        Args:
            model_path (str): Path to the trained model file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found.")
                return False
                
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Extract embeddings from the model
            try:
                # Get product embeddings
                product_embedding_layer = self.model.get_layer('product_embedding')
                self.product_embeddings = product_embedding_layer.get_weights()[0]
                
                # Get user embeddings
                user_embedding_layer = self.model.get_layer('user_embedding')
                self.user_embeddings = user_embedding_layer.get_weights()[0]
                
                print(f"Embeddings extracted: Users: {self.user_embeddings.shape}, Products: {self.product_embeddings.shape}")
            except Exception as e:
                print(f"Error extracting embeddings: {e}")
                self.product_embeddings = None
                self.user_embeddings = None
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_metadata(self, metadata_path):
        """
        Load metadata for the recommendation model.
        
        Args:
            metadata_path (str): Path to the metadata pickle file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(metadata_path):
                print(f"Metadata file {metadata_path} not found.")
                return False
                
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Extract components from metadata
            self.product_encoder = self.metadata.get('product_encoder')
            self.idx_to_product = self.metadata.get('idx_to_product')
            self.product_to_name = self.metadata.get('product_to_name')
            self.rating_scaler = self.metadata.get('rating_scaler')
            self.category_mapping = self.metadata.get('category_mapping')
            
            # Use embeddings from metadata if available and not already loaded
            if self.product_embeddings is None and 'product_embeddings' in self.metadata:
                self.product_embeddings = self.metadata.get('product_embeddings')
                
            if self.user_embeddings is None and 'user_embeddings' in self.metadata:
                self.user_embeddings = self.metadata.get('user_embeddings')
            
            print(f"Metadata loaded successfully from {metadata_path}")
            return True
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return False
    
    def load_product_data(self, csv_path='amazon_cleaned.csv'):
        """
        Load product data from CSV file.
        
        Args:
            csv_path (str): Path to the product CSV file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(csv_path):
                print(f"CSV file {csv_path} not found.")
                return False
                
            self.df = pd.read_csv(csv_path)
            print(f"Product data loaded successfully: {len(self.df)} products")
            return True
        except Exception as e:
            print(f"Error loading product data: {e}")
            return False
    
    def set_category_weight(self, weight):
        """
        Set the weight for category bonus in recommendations.
        
        Args:
            weight (float): Weight value between 0 and 1
        """
        if 0 <= weight <= 1:
            self.category_weight = weight
            print(f"Category weight set to {weight}")
        else:
            print("Category weight must be between 0 and 1")
    
    def get_similar_products(self, product_id, n=5, category_bonus=True):
        """
        Get similar products based on embedding similarity with category bonus.
        
        Args:
            product_id (str): The product ID to find similar products for
            n (int): Number of similar products to return
            category_bonus (bool): Whether to apply category bonus
        
        Returns:
            list: List of similar product dictionaries
        """
        if self.product_embeddings is None or self.product_encoder is None:
            print("Product embeddings or encoder not available")
            return []
        
        try:
            # Get product index
            product_idx = self.product_encoder.transform([product_id])[0]
            
            # Get product embedding
            product_embedding = self.product_embeddings[product_idx]
            
            # Get product category if available
            product_category = None
            if self.category_mapping and product_idx in self.category_mapping:
                product_category = self.category_mapping[product_idx]
            
            # Calculate similarity with all products
            similarities = []
            for idx, embedding in enumerate(self.product_embeddings):
                if idx == product_idx:  # Skip the same product
                    continue
                
                # Calculate cosine similarity
                dot_product = np.dot(product_embedding, embedding)
                norm_product1 = np.linalg.norm(product_embedding)
                norm_product2 = np.linalg.norm(embedding)
                
                if norm_product1 == 0 or norm_product2 == 0:
                    cosine_sim = 0
                else:
                    cosine_sim = dot_product / (norm_product1 * norm_product2)
                
                # Apply category bonus if enabled
                category_bonus_value = 0
                if category_bonus and product_category and self.category_mapping:
                    if idx in self.category_mapping and self.category_mapping[idx] == product_category:
                        category_bonus_value = self.category_weight
                
                # Final score combines similarity and category
                final_score = ((1 - self.category_weight) * cosine_sim) + category_bonus_value
                
                similarities.append((idx, final_score, cosine_sim))
            
            # Sort by final score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top similar products
            similar_products = []
            for idx, final_score, cosine_sim in similarities[:n]:
                try:
                    product_id = self.idx_to_product[idx]
                    product_name = self.product_to_name.get(product_id, "Unknown")
                    
                    category = None
                    if self.category_mapping and idx in self.category_mapping:
                        category = self.category_mapping[idx]
                    
                    similar_products.append({
                        'product_id': product_id,
                        'name': product_name,
                        'similarity': cosine_sim,
                        'final_score': final_score,
                        'category': category
                    })
                except Exception as e:
                    print(f"Error processing product {idx}: {e}")
                    continue
            
            return similar_products
        
        except Exception as e:
            print(f"Error finding similar products: {e}")
            return []
    
    def recommend_for_user(self, user_idx, n=5, recent_category=None):
        """
        Get personalized recommendations for a user with category-based enhancement.
        
        Args:
            user_idx (int): User index
            n (int): Number of recommendations to return
            recent_category (str): User's recently viewed category
        
        Returns:
            list: List of recommended product dictionaries
        """
        if self.model is None or self.user_embeddings is None:
            print("Model or user embeddings not available")
            return []
        
        try:
            # Get user embedding
            user_embedding = self.user_embeddings[user_idx]
            
            # Calculate similarity with all products
            similarities = []
            for idx, embedding in enumerate(self.product_embeddings):
                # Calculate cosine similarity
                dot_product = np.dot(user_embedding, embedding)
                norm_user = np.linalg.norm(user_embedding)
                norm_product = np.linalg.norm(embedding)
                
                if norm_user == 0 or norm_product == 0:
                    cosine_sim = 0
                else:
                    cosine_sim = dot_product / (norm_user * norm_product)
                
                # Apply category bonus if recent category is provided
                category_bonus = 0
                if recent_category and self.category_mapping:
                    if idx in self.category_mapping and self.category_mapping[idx] == recent_category:
                        category_bonus = self.category_weight
                
                # Final score combines similarity and category
                final_score = ((1 - self.category_weight) * cosine_sim) + category_bonus
                
                similarities.append((idx, final_score, cosine_sim))
            
            # Sort by final score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top recommended products
            recommendations = []
            for idx, final_score, cosine_sim in similarities[:n]:
                try:
                    product_id = self.idx_to_product[idx]
                    product_name = self.product_to_name.get(product_id, "Unknown")
                    
                    category = None
                    if self.category_mapping and idx in self.category_mapping:
                        category = self.category_mapping[idx]
                    
                    recommendations.append({
                        'product_id': product_id,
                        'name': product_name,
                        'similarity': cosine_sim,
                        'final_score': final_score,
                        'category': category
                    })
                except Exception as e:
                    print(f"Error processing product {idx}: {e}")
                    continue
            
            return recommendations
        
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
    
    def recommend_from_cart(self, cart_items, n=5, recent_category=None):
        """
        Get recommendations based on items in the cart with category enhancement.
        
        Args:
            cart_items (list): List of product IDs in the cart
            n (int): Number of recommendations to return
            recent_category (str): User's recently viewed category
        
        Returns:
            list: List of recommended product dictionaries
        """
        if self.product_embeddings is None or self.product_encoder is None:
            print("Product embeddings or encoder not available")
            return []
        
        try:
            # Get product indices for cart items
            cart_indices = []
            for product_id in cart_items:
                try:
                    product_idx = self.product_encoder.transform([product_id])[0]
                    if product_idx < len(self.product_embeddings):
                        cart_indices.append(product_idx)
                except:
                    continue
            
            if not cart_indices:
                return []
            
            # Create virtual user embedding as average of cart product embeddings
            cart_embeddings = [self.product_embeddings[idx] for idx in cart_indices]
            virtual_user_embedding = np.mean(cart_embeddings, axis=0)
            
            # Calculate similarity with all products
            similarities = []
            for idx, embedding in enumerate(self.product_embeddings):
                # Skip products already in cart
                if idx in cart_indices:
                    continue
                
                # Calculate cosine similarity
                dot_product = np.dot(virtual_user_embedding, embedding)
                norm_user = np.linalg.norm(virtual_user_embedding)
                norm_product = np.linalg.norm(embedding)
                
                if norm_user == 0 or norm_product == 0:
                    cosine_sim = 0
                else:
                    cosine_sim = dot_product / (norm_user * norm_product)
                
                # Apply category bonus if recent category is provided
                category_bonus = 0
                if recent_category and self.category_mapping:
                    if idx in self.category_mapping and self.category_mapping[idx] == recent_category:
                        category_bonus = self.category_weight
                
                # Final score combines similarity and category
                final_score = ((1 - self.category_weight) * cosine_sim) + category_bonus
                
                similarities.append((idx, final_score, cosine_sim))
            
            # Sort by final score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top recommended products
            recommendations = []
            for idx, final_score, cosine_sim in similarities[:n]:
                try:
                    product_id = self.idx_to_product[idx]
                    product_name = self.product_to_name.get(product_id, "Unknown")
                    
                    category = None
                    if self.category_mapping and idx in self.category_mapping:
                        category = self.category_mapping[idx]
                    
                    recommendations.append({
                        'product_id': product_id,
                        'name': product_name,
                        'similarity': cosine_sim,
                        'final_score': final_score,
                        'category': category
                    })
                except Exception as e:
                    print(f"Error processing product {idx}: {e}")
                    continue
            
            return recommendations
        
        except Exception as e:
            print(f"Error generating recommendations from cart: {e}")
            return []
    
    def predict_rating(self, user_idx, product_id):
        """
        Predict rating for a user-product pair.
        
        Args:
            user_idx (int): User index
            product_id (str): Product ID
        
        Returns:
            float: Predicted rating (1-5 scale)
        """
        if self.model is None or self.rating_scaler is None:
            print("Model or rating scaler not available")
            return 0
        
        try:
            # Convert product ID to index
            product_idx = self.product_encoder.transform([product_id])[0]
            
            # Prepare input
            user_input = np.array([[user_idx]])
            product_input = np.array([[product_idx]])
            
            # Make prediction
            prediction = self.model.predict([user_input, product_input], verbose=0)
            
            # Convert from [0,1] scale to original rating scale
            original_rating = self.rating_scaler.inverse_transform(prediction)[0][0]
            
            return original_rating
        
        except Exception as e:
            print(f"Error predicting rating: {e}")
        return 0
    def get_products_by_category(self, category, limit=10):
        """
        Get products from a specific category
        
        Args:
            category (str): Category name
            limit (int): Maximum number of products to return
            
        Returns:
            list: List of product dictionaries in the category
        """
        if self.df is None:
            print("Product data not available")
            return []
        
        try:
            # Filter by category
            category_products = self.df[self.df['category'] == category]
            
            # Sort by popularity if available
            if 'popularity' in self.df.columns:
                category_products = category_products.sort_values('popularity', ascending=False)
            
            # Get top products
            result = []
            for _, product in category_products.head(limit).iterrows():
                result.append({
                    'product_id': product['product_id'],
                    'name': product['product_name'],
                    'price': product.get('actual_price', 'N/A'),
                    'discounted_price': product.get('discounted_price', 'N/A'),
                    'rating': product.get('rating', 'N/A'),
                    'image_url': product.get('img_link', '/static/img/placeholder.png')
                })
            
            return result
        
        except Exception as e:
            print(f"Error getting products by category: {e}")
            return []
    
    def get_category_bonus_performance(self, test_size=50):
        """
        Test the performance impact of different category bonus weights
        
        Args:
            test_size (int): Number of test cases
            
        Returns:
            dict: Performance metrics
        """
        if not self.product_embeddings or not self.category_mapping:
            return {"error": "Product embeddings or category mapping not available"}
        
        # Sample product indices
        import random
        product_indices = random.sample(range(len(self.product_embeddings)), min(test_size, len(self.product_embeddings)))
        
        results = []
        weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        for weight in weights:
            # Set weight for testing
            self.set_category_weight(weight)
            
            same_category_count = 0
            total_recommendations = 0
            
            for idx in product_indices:
                # Skip products without category
                if not (self.category_mapping and idx in self.category_mapping):
                    continue
                    
                # Get original product category
                original_category = self.category_mapping[idx]
                
                # Get product ID
                try:
                    product_id = self.idx_to_product[idx]
                    
                    # Get similar products
                    similar_products = self.get_similar_products(product_id, n=5, category_bonus=True)
                    
                    # Count recommendations in same category
                    for product in similar_products:
                        total_recommendations += 1
                        if product.get('category') == original_category:
                            same_category_count += 1
                except:
                    continue
            
            # Calculate percentage
            if total_recommendations > 0:
                same_category_percentage = (same_category_count / total_recommendations) * 100
            else:
                same_category_percentage = 0
                
            results.append({
                'weight': weight,
                'same_category_percentage': same_category_percentage,
                'total_recommendations': total_recommendations
            })
        
        # Reset to default weight
        self.set_category_weight(0.6)
        
        return results
    