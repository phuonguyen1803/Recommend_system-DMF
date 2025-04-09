"""
Utility module for the Amazon Recommendation System.
This module provides helper functions for loading models and data, 
processing products, and generating recommendations.
"""

import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from typing import Dict, List, Any, Union, Optional, Tuple

def load_model_and_data():
    """
    Load the trained recommendation model, metadata, and product data.
    
    Returns:
        dict: Dictionary containing loaded model, metadata, and data
    """
    print("Loading model and data...")
    result = {}
    
    # Load the model
    try:
        model = tf.keras.models.load_model('amazon_recommender_model.keras')
        print("Model loaded successfully")
        result['model'] = model
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        result['model'] = None
    
    # Load metadata
    try:
        with open('amazon_recommender_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        result['metadata'] = metadata
        
        if metadata is not None:
            # Extract components from metadata
            product_encoder = metadata.get('product_encoder')
            user_encoder = metadata.get('user_encoder')
            idx_to_product = metadata.get('idx_to_product')
            product_to_name = metadata.get('product_to_name')
            rating_scaler = metadata.get('rating_scaler')
            category_mapping = metadata.get('category_mapping')
            
            # Extract embeddings if available
            user_embeddings = metadata.get('user_embeddings')
            product_embeddings = metadata.get('product_embeddings')
            
            result['product_encoder'] = product_encoder
            result['user_encoder'] = user_encoder 
            result['idx_to_product'] = idx_to_product
            result['product_to_name'] = product_to_name
            result['rating_scaler'] = rating_scaler
            result['category_mapping'] = category_mapping
            result['user_embeddings'] = user_embeddings
            result['product_embeddings'] = product_embeddings
            
            print("Metadata loaded successfully")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = None
        result['metadata'] = None
    
    # Load product data
    try:
        df = pd.read_csv('amazon_cleaned.csv')
        # Sort categories for display
        if 'category' in df.columns:
            product_categories = sorted(df['category'].dropna().unique().tolist())
        else:
            product_categories = []
        
        result['df'] = df
        result['product_categories'] = product_categories
        print(f"Data loaded successfully: {len(df)} products, {len(product_categories)} categories")
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        df = None
        result['df'] = None
        result['product_categories'] = []

    return result

def get_product_details(df: pd.DataFrame, product_id: str) -> Optional[Dict[str, Any]]:
    """
    Get product details from the dataframe
    
    Args:
        df: DataFrame containing product data
        product_id: ID of the product to retrieve
        
    Returns:
        dict: Product details or None if not found
    """
    if df is None:
        return None
    
    product_data = df[df['product_id'] == product_id]
    if product_data.empty:
        return None
    
    product = product_data.iloc[0]
    
    details = {
        'product_id': product['product_id'],
        'name': product['product_name'],
        'category': product.get('category', 'Uncategorized'),
        'price': product.get('actual_price', 'N/A'),
        'discounted_price': product.get('discounted_price', 'N/A'),
        'discount_percentage': product.get('discount_percentage', '0%'),
        'rating': product.get('rating', 'N/A'),
        'rating_count': product.get('rating_count', '0'),
        'about': product.get('about_product', 'No description available'),
        'image_url': product.get('img_link', '/static/img/placeholder.png')
    }
    
    return details

def get_popular_products(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get popular products based on ratings and rating count
    
    Args:
        df: DataFrame containing product data
        limit: Maximum number of products to return
        
    Returns:
        list: List of popular product dictionaries
    """
    if df is None:
        return []
    
    try:
        # Convert columns to numeric
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating_count_numeric'] = pd.to_numeric(df['rating_count'], errors='coerce')
        
        # Filter products with at least some ratings
        popular = df[df['rating_count_numeric'] > 10]
        
        # Calculate popularity score
        popular['popularity'] = popular['rating_numeric'] * np.log1p(popular['rating_count_numeric'])
        
        # Sort by popularity score
        popular = popular.sort_values(by='popularity', ascending=False)
        
        # Get top products
        top_products = []
        for _, product in popular.head(limit).iterrows():
            top_products.append({
                'product_id': product['product_id'],
                'name': product['product_name'],
                'price': product.get('actual_price', 'N/A'),
                'discounted_price': product.get('discounted_price', 'N/A'),
                'rating': product.get('rating', 'N/A'),
                'image_url': product.get('img_link', '/static/img/placeholder.png'),
                'category': product.get('category', 'Uncategorized')
            })
        
        return top_products
    except Exception as e:
        print(f"Error getting popular products: {e}")
        return []

def get_products_by_category(df: pd.DataFrame, category: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get products by category
    
    Args:
        df: DataFrame containing product data
        category: Category name
        limit: Maximum number of products to return
        
    Returns:
        list: List of product dictionaries in the category
    """
    if df is None:
        return []
    
    category_products = df[df['category'] == category]
    
    # Sort by popularity if available
    if 'rating' in df.columns and 'rating_count' in df.columns:
        category_products['rating_numeric'] = pd.to_numeric(category_products['rating'], errors='coerce')
        category_products['rating_count_numeric'] = pd.to_numeric(category_products['rating_count'], errors='coerce')
        category_products['popularity'] = category_products['rating_numeric'] * np.log1p(category_products['rating_count_numeric'])
        category_products = category_products.sort_values('popularity', ascending=False)
    
    products = []
    for _, product in category_products.head(limit).iterrows():
        products.append({
            'product_id': product['product_id'],
            'name': product['product_name'],
            'price': product.get('actual_price', 'N/A'),
            'discounted_price': product.get('discounted_price', 'N/A'),
            'rating': product.get('rating', 'N/A'),
            'image_url': product.get('img_link', '/static/img/placeholder.png'),
            'category': product.get('category', 'Uncategorized')
        })
    
    return products

def get_recommended_products(model, metadata, df, cart_items, limit=5):
    """
    Get product recommendations based on items in cart with category enhancement
    
    Args:
        model: Trained recommendation model
        metadata: Model metadata dictionary
        df: DataFrame containing product data
        cart_items: List of product IDs in the cart
        limit: Maximum number of recommendations to return
        
    Returns:
        list: List of recommended product dictionaries
    """
    if model is None or metadata is None or df is None:
        return get_popular_products(df, limit)
    
    try:
        # Get embeddings
        product_embeddings = None
        if 'product_embeddings' in metadata and metadata['product_embeddings'] is not None:
            product_embeddings = metadata['product_embeddings']
        else:
            product_embedding_layer = model.get_layer('product_embedding')
            product_embeddings = product_embedding_layer.get_weights()[0]
        
        # Get necessary data from metadata
        product_encoder = metadata.get('product_encoder')
        idx_to_product = metadata.get('idx_to_product')
        category_mapping = metadata.get('category_mapping')
        
        # Convert cart items to indices
        cart_product_indices = []
        for product_id in cart_items:
            try:
                product_idx = product_encoder.transform([product_id])[0]
                if product_idx < len(product_embeddings):
                    cart_product_indices.append(product_idx)
            except:
                try:
                    product_data = df[df['product_id'] == product_id]
                    if not product_data.empty:
                        if 'product_idx' in product_data.columns:
                            product_idx = int(product_data.iloc[0]['product_idx'])
                            if product_idx < len(product_embeddings):
                                cart_product_indices.append(product_idx)
                except:
                    pass
        
        if not cart_product_indices:
            return get_popular_products(df, limit)
        
        # Create virtual user embedding as average of cart product embeddings
        cart_embeddings = [product_embeddings[idx] for idx in cart_product_indices]
        virtual_user_embedding = np.mean(cart_embeddings, axis=0)
        
        # Get recently viewed category (if in session)
        from flask import session
        recent_category = None
        recent_view = session.get('recent_view') if 'session' in globals() else None
        if recent_view:
            try:
                recent_product_data = df[df['product_id'] == recent_view]
                if not recent_product_data.empty and 'category' in recent_product_data.columns:
                    recent_category = recent_product_data.iloc[0]['category']
            except:
                pass
        
        # Calculate similarity with all products, adding category bonus
        similarities = []
        category_weight = 0.6  # Weight for category bonus
        
        for idx, product_embedding in enumerate(product_embeddings):
            # Skip products already in cart
            if idx in cart_product_indices:
                continue
                
            # Calculate cosine similarity
            dot_product = np.dot(virtual_user_embedding, product_embedding)
            norm_user = np.linalg.norm(virtual_user_embedding)
            norm_product = np.linalg.norm(product_embedding)
            
            if norm_user == 0 or norm_product == 0:
                cosine_sim = 0
            else:
                cosine_sim = dot_product / (norm_user * norm_product)
            
            # Calculate category bonus
            category_bonus = 0
            if recent_category and category_mapping and idx in category_mapping:
                if category_mapping[idx] == recent_category:
                    category_bonus = category_weight
            
            # Final score combines similarity and category
            final_score = ((1 - category_weight) * cosine_sim) + category_bonus
            
            similarities.append((idx, final_score))
        
        # Sort by final score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommendations = []
        for idx, _ in similarities[:limit]:
            try:
                product_id = idx_to_product.get(idx) or product_encoder.inverse_transform([idx])[0]
                product_details = get_product_details(df, product_id)
                if product_details:
                    recommendations.append(product_details)
            except Exception as e:
                print(f"Error getting product details for index {idx}: {e}")
                continue
        
        return recommendations
    except Exception as e:
        import traceback
        print(f"Error generating recommendations: {e}")
        traceback.print_exc()
        return get_popular_products(df, limit)

def search_products(df, query, limit=20):
    """
    Search products by name
    
    Args:
        df: DataFrame containing product data
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        list: List of matching product dictionaries
    """
    if df is None:
        return []
    
    query = query.lower()
    
    try:
        # Search in product name
        name_matches = df[df['product_name'].str.lower().str.contains(query, na=False)]
        
        # Also search in description if available
        desc_matches = pd.DataFrame()
        if 'about_product' in df.columns:
            desc_matches = df[df['about_product'].str.lower().str.contains(query, na=False)]
        
        # Combine results, removing duplicates
        matching_products = pd.concat([name_matches, desc_matches]).drop_duplicates(subset=['product_id'])
        
        # Sort by relevance (if name contains query, it's more relevant)
        name_relevance = matching_products['product_name'].str.lower().str.contains(f"\\b{query}\\b", regex=True).astype(int)
        matching_products['relevance'] = name_relevance
        
        # Also sort by popularity if rating data is available
        if 'rating' in matching_products.columns and 'rating_count' in matching_products.columns:
            matching_products['rating_numeric'] = pd.to_numeric(matching_products['rating'], errors='coerce')
            matching_products['rating_count_numeric'] = pd.to_numeric(matching_products['rating_count'], errors='coerce')
            matching_products['popularity'] = matching_products['rating_numeric'] * np.log1p(matching_products['rating_count_numeric'])
            matching_products = matching_products.sort_values(['relevance', 'popularity'], ascending=[False, False])
        else:
            matching_products = matching_products.sort_values('relevance', ascending=False)
        
        # Get results
        results = []
        for _, product in matching_products.head(limit).iterrows():
            results.append({
                'product_id': product['product_id'],
                'name': product['product_name'],
                'price': product.get('actual_price', 'N/A'),
                'discounted_price': product.get('discounted_price', 'N/A'),
                'rating': product.get('rating', 'N/A'),
                'image_url': product.get('img_link', '/static/img/placeholder.png'),
                'category': product.get('category', 'Uncategorized')
            })
        
        return results
    except Exception as e:
        print(f"Error searching products: {e}")
        return []