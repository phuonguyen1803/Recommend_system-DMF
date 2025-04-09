import random
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from utils.data_processing import process_and_save_data
from data.database import (
    signup_user,
    login_user,
    save_browsing_history,
    update_cart_history,
    update_view_history,
    save_search_history,
    products_collection,
    users_collection,
    browsing_history_collection
)
from datetime import datetime
import threading
from functools import lru_cache
from cachelib import SimpleCache

app = Flask(__name__)
app.secret_key = '12345'

# Configuration
DEBUG_MODE = True
DEVELOPMENT_MODE = False  # Set to True to load only a small sample of data
CACHE_TIMEOUT = 300       # Seconds

# Cache for Flask
cache = SimpleCache()

# Global variables
model = None
metadata = None
df = None
product_encoder = None
idx_to_product = None
product_to_name = None
rating_scaler = None
product_categories = None
category_mapping = None
user_embeddings = None
product_embeddings = None

# State tracking
model_loaded = False
metadata_loaded = False
df_loaded = False

# ----------------- Helper Functions to Persist Cart Categories -----------------
def save_cart_categories(user_id, cart):
    """
    Save the categories of current cart items to a JSON file.
    File name: cart_categories_{user_id}.json
    """
    categories = set()
    for pid in cart.keys():
        prod = get_product_details(pid)
        if prod and prod.get('category'):
            categories.add(prod['category'])
    filename = f"cart_categories_{user_id}.json"
    try:
        with open(filename, "w") as f:
            json.dump(list(categories), f)
        print(f"DEBUG: Saved cart categories for user {user_id}: {list(categories)}")
    except Exception as e:
        print(f"DEBUG: Error saving cart categories for user {user_id}: {e}")

def load_cart_categories(user_id):
    """
    Load saved cart categories for a user from a JSON file.
    Returns a set of categories.
    """
    filename = f"cart_categories_{user_id}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                categories = json.load(f)
            print(f"DEBUG: Loaded cart categories for user {user_id}: {categories}")
            return set(categories)
        except Exception as e:
            print(f"DEBUG: Error loading cart categories for user {user_id}: {e}")
    else:
        print(f"DEBUG: No saved cart categories found for user {user_id}")
    return set()

# ----------------- Loading Functions -----------------
def load_model():
    global model, model_loaded
    if model_loaded:
        return model
    try:
        print("DEBUG: Loading model...")
        model = tf.keras.models.load_model('amazon_recommender_model_updated.keras')
        print("DEBUG: Model loaded successfully from amazon_recommender_model_updated.keras")
        model_loaded = True
        return model
    except Exception as e:
        print(f"DEBUG: Error loading model: {e}")
        return None

def load_metadata():
    global metadata, metadata_loaded, product_encoder, idx_to_product, product_to_name
    global rating_scaler, category_mapping, user_embeddings, product_embeddings
    if metadata_loaded:
        return metadata
    try:
        print("DEBUG: Loading metadata...")
        with open('amazon_recommender_metadata_updated.pkl', 'rb') as f:
            metadata = pickle.load(f)
        print("DEBUG: Metadata loaded successfully")
        if metadata is not None:
            product_encoder = metadata.get('product_encoder')
            user_encoder = metadata.get('user_encoder', None)
            idx_to_product = metadata.get('idx_to_product')
            product_to_name = metadata.get('product_to_name')
            rating_scaler = metadata.get('rating_scaler')
            category_mapping = metadata.get('category_mapping')
            user_embeddings = metadata.get('user_embeddings')
            product_embeddings = metadata.get('product_embeddings')
        metadata_loaded = True
        return metadata
    except Exception as e:
        print(f"DEBUG: Error loading metadata: {e}")
        return None

def load_dataframe(nrows=None):
    global df, df_loaded, product_categories
    if df_loaded and nrows is None:
        return df
    try:
        if os.path.exists('amazon_cleaned.csv'):
            print(f"DEBUG: Loading CSV data{' (limited rows)' if nrows else ''}...")
            df = pd.read_csv('amazon_cleaned.csv', nrows=nrows)
            print(f"DEBUG: Data loaded; total products: {len(df)}")
        else:
            print("DEBUG: CSV data file not found")
            df = None
        if df is not None and 'category' in df.columns:
            product_categories = sorted(df['category'].dropna().unique().tolist())
        else:
            product_categories = []
        if nrows is None:
            df_loaded = True
        return df
    except Exception as e:
        print(f"DEBUG: Error loading CSV data: {e}")
        return None

def load_in_background():
    threading.Thread(target=load_model).start()
    threading.Thread(target=load_metadata).start()
    threading.Thread(target=lambda: load_dataframe(None)).start()

def load_initial_data():
    if DEVELOPMENT_MODE:
        return load_dataframe(nrows=None)
    else:
        sample_df = load_dataframe(nrows=None)
        load_in_background()
        return sample_df

# ------------------ Cached Functions ---------------------
@lru_cache(maxsize=128)
def get_top_products(limit=5):
    cache_key = f'top_products_{limit}'
    cached_result = cache.get(cache_key)
    if cached_result:
        print("DEBUG: Returning cached top products")
        return cached_result
    if products_collection is None:
        result = get_popular_products(get_dataframe(), limit)
    else:
        try:
            top_products = list(products_collection.find().sort('popularity', -1).limit(limit))
            result = [
                {
                    'product_id': product['product_id'],
                    'name': product['product_name'],
                    'image_url': product.get('img_link', '/static/img/placeholder.png'),
                    'discounted_price': product.get('discounted_price', 'N/A'),
                    'price': product.get('actual_price', 'N/A'),
                    'rating': product.get('rating', 'N/A'),
                    'popularity': product.get('popularity', 0),
                    'category': product.get('category', 'Uncategorized')
                }
                for product in top_products
            ]
            print("DEBUG: Top products from MongoDB:", result)
        except Exception as e:
            print(f"DEBUG: Error getting top products from MongoDB: {e}")
            result = get_popular_products(get_dataframe(), limit)
    cache.set(cache_key, result, timeout=CACHE_TIMEOUT)
    return result

def get_dataframe():
    global df
    if df is None:
        load_initial_data()
    return df

@lru_cache(maxsize=256)
def get_product_details(product_id):
    cache_key = f'product_{product_id}'
    cached_result = cache.get(cache_key)
    if cached_result:
        print(f"DEBUG: Returning cached details for product: {product_id}")
        return cached_result
    if products_collection is not None:
        try:
            product = products_collection.find_one({'product_id': product_id})
            if product:
                result = {
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
                cache.set(cache_key, result, timeout=CACHE_TIMEOUT)
                print(f"DEBUG: Found product {product_id} in MongoDB")
                return result
        except Exception as e:
            print(f"DEBUG: Error getting product from MongoDB: {e}")
    current_df = get_dataframe()
    if current_df is None:
        return None
    product_data = current_df[current_df['product_id'] == product_id]
    if product_data.empty:
        return None
    product = product_data.iloc[0]
    result = {
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
    cache.set(cache_key, result, timeout=CACHE_TIMEOUT)
    print(f"DEBUG: Returning product details from DataFrame for {product_id}")
    return result

def get_popular_products(df, limit=10):
    cache_key = f'popular_products_{limit}'
    cached_result = cache.get(cache_key)
    if cached_result:
        print("DEBUG: Returning cached popular products")
        return cached_result
    if df is None:
        return []
    try:
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        popular = df[pd.to_numeric(df['rating_count'], errors='coerce') > 10]
        popular = popular.sort_values(by='rating_numeric', ascending=False)
        top_products = []
        unique_products = set()
        for _, product in popular.iterrows():
            if product['product_id'] not in unique_products:
                unique_products.add(product['product_id'])
                top_products.append({
                    'product_id': product['product_id'],
                    'name': product['product_name'],
                    'price': product.get('actual_price', 'N/A'),
                    'discounted_price': product.get('discounted_price', 'N/A'),
                    'rating': product.get('rating', 'N/A'),
                    'image_url': product.get('img_link', '/static/img/placeholder.png'),
                    'category': product.get('category', 'Uncategorized')
                })
            if len(top_products) >= limit:
                break
        cache.set(cache_key, top_products, timeout=CACHE_TIMEOUT)
        print("DEBUG: Popular products computed:", top_products)
        return top_products
    except Exception as e:
        print(f"DEBUG: Error getting popular products: {e}")
        return []

def get_products_by_category(category, limit=20):
    cache_key = f'category_{category}_{limit}'
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    current_df = get_dataframe()
    if products_collection is not None:
        try:
            offset = 0
            products = []
            unique_product_ids = set()
            while True:
                category_products = list(products_collection.find({'category': category}).skip(offset).limit(limit))
                print(f"DEBUG: Found {len(category_products)} products for category {category}")
                if not category_products:
                    break
                for product in category_products:
                    if product['product_id'] not in unique_product_ids:
                        unique_product_ids.add(product['product_id'])
                        products.append({
                            'product_id': product['product_id'],
                            'name': product['product_name'],
                            'price': product.get('actual_price', 'N/A'),
                            'discounted_price': product.get('discounted_price', 'N/A'),
                            'rating': product.get('rating', 'N/A'),
                            'image_url': product.get('img_link', '/static/img/placeholder.png'),
                            'category': product.get('category', 'Uncategorized')
                        })
                if len(products) >= limit:
                    cache.set(cache_key, products, timeout=CACHE_TIMEOUT)
                    return products
                offset += limit
        except Exception as e:
            print(f"DEBUG: Error getting products by category from MongoDB: {e}")
    if current_df is None:
        return []
    category_products = current_df[current_df['category'] == category]
    products = []
    unique_product_ids = set()
    for _, product in category_products.iterrows():
        if product['product_id'] not in unique_product_ids:
            unique_product_ids.add(product['product_id'])
            products.append({
                'product_id': product['product_id'],
                'name': product['product_name'],
                'price': product.get('actual_price', 'N/A'),
                'discounted_price': product.get('discounted_price', 'N/A'),
                'rating': product.get('rating', 'N/A'),
                'image_url': product.get('img_link', '/static/img/placeholder.png'),
                'category': product.get('category', 'Uncategorized')
            })
        if len(products) >= limit:
            break
    cache.set(cache_key, products, timeout=CACHE_TIMEOUT)
    return products
# ----- Updated Recommendation Function Using Categories -----
def get_recommended_products(cart_items=None, user_id=None, limit=5):

    return None
# ----- New Recommendation Function Reading Cart Categories from File -----
def get_recommended_products_from_file(user_id, limit=5):
    print("DEBUG: get_recommended_products_from_file called for user:", user_id)
    df = get_dataframe()
    saved = load_cart_categories(user_id)  # e.g. ['Books', 'Electronics', …]
    if not saved:
        return get_popular_products(df, limit)

    from collections import Counter
    freq = Counter(saved) 
    recency = {cat: idx for idx, cat in enumerate(saved)}

    # avoid zero-division
    max_f = max(freq.values()) or 1
    max_r = max(recency.values()) or 1
    # compute weights
    cat_weights = {}
    for cat in freq:
        f = freq[cat] / max_f  
        r = recency[cat] / max_r
        cat_weights[cat] = 0.4 * r + 0.6 * f
        print(f"DEBUG: cat={cat}, freq={freq[cat]}, recency={recency[cat]}, weight={cat_weights[cat]:.2f}")

    # build candidates
    candidates = []
    for cat, w in cat_weights.items():
        cap = max(1, int(round(limit * w)))
        prods = get_products_by_category(cat, limit=cap)
        print(f"DEBUG: Fetching up to {cap} items for category '{cat}'")
        for p in prods:
            try:
                pop = float(p.get('rating', 0)) * np.log1p(float(p.get('rating_count', 0)))
                pop_norm = pop / (1 + pop)
            except:
                pop_norm = 0
            candidates.append((p, w, pop_norm))

    # score & dedupe
    scored = {}
    for p, w, popn in candidates:
        score = 0.5 * w + 0.4 * popn + 0.1 * random.random()
        pid = p['product_id']
        if pid not in scored or score > scored[pid][1]:
            scored[pid] = (p, score)

    # sort & trim
    recs = [prod for prod, _ in sorted(scored.values(), key=lambda x: x[1], reverse=True)]
    if len(recs) < limit:
        print("DEBUG: Not enough from categories, filling with popular")
        recs += get_popular_products(df, limit)
    final = recs[:limit]
    print("DEBUG: Final recs:", [p['product_id'] for p in final])
    return final



def search_products(query, limit=20):
    cache_key = f'search_{query}_{limit}'
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    current_df = get_dataframe()
    if products_collection is not None:
        try:
            offset = 0
            results = []
            unique_product_ids = set()
            while True:
                search_query = {'product_name': {'$regex': query, '$options': 'i'}}
                search_results = list(products_collection.find(search_query).skip(offset).limit(limit))
                if not search_results:
                    break
                for product in search_results:
                    if product['product_id'] not in unique_product_ids:
                        unique_product_ids.add(product['product_id'])
                        results.append({
                            'product_id': product['product_id'],
                            'name': product['product_name'],
                            'price': product.get('actual_price', 'N/A'),
                            'discounted_price': product.get('discounted_price', 'N/A'),
                            'rating': product.get('rating', 'N/A'),
                            'image_url': product.get('img_link', '/static/img/placeholder.png'),
                            'category': product.get('category', 'Uncategorized')
                        })
                if len(results) >= limit:
                    cache.set(cache_key, results, timeout=CACHE_TIMEOUT)
                    return results
                offset += limit
        except Exception as e:
            print(f"DEBUG: Error searching products in MongoDB: {e}")
    if current_df is None:
        return []
    query = query.lower()
    matching_products = current_df[current_df['product_name'].str.lower().str.contains(query, na=False)]
    results = []
    unique_product_ids = set()
    for _, product in matching_products.iterrows():
        if product['product_id'] not in unique_product_ids:
            unique_product_ids.add(product['product_id'])
            results.append({
                'product_id': product['product_id'],
                'name': product['product_name'],
                'price': product.get('actual_price', 'N/A'),
                'discounted_price': product.get('discounted_price', 'N/A'),
                'rating': product.get('rating', 'N/A'),
                'image_url': product.get('img_link', '/static/img/placeholder.png'),
                'category': product.get('category', 'Uncategorized')
            })
            if len(results) >= limit:
                break
    cache.set(cache_key, results, timeout=CACHE_TIMEOUT)
    return results

# ----------------------- Routes ---------------------------
@app.route('/')
def home():
    print("DEBUG: Rendering home page")
    top_products = get_top_products(limit=5)
    popular_products = get_popular_products(get_dataframe(), 8)
    if product_categories and len(product_categories) > 0:
        featured_category = random.choice(product_categories)
        featured_products = get_products_by_category(featured_category, 4)
    else:
        featured_category = "Featured Products"
        featured_products = popular_products[:4] if popular_products else []
    cart = session.get('cart', {})
    current_user = session.get('user')
    # For this example, if the user is logged in, we use the file-based recommendations.
    if current_user:
        recommended_products = get_recommended_products_from_file(current_user.get('email'), limit=4)
    else:
        recommended_products = get_recommended_products(list(cart.keys()), None, limit=4)
    print("DEBUG: Homepage recommended_products:", recommended_products)
    response = render_template('home.html', 
                               top_products=top_products,
                               popular_products=popular_products,
                               featured_category=featured_category,
                               featured_products=featured_products,
                               recommended_products=recommended_products,
                               categories=product_categories,
                               current_user=current_user)
    return response

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('auth/signup.html', categories=product_categories)
    elif request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
        else:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
        success, message = signup_user(username, email, password)
        if success:
            flash('Sign up successful! Please login.', 'success')
            if request.is_json:
                return jsonify({'message': message, 'success': True}), 200
            return redirect(url_for('login'))
        flash(message, 'danger')
        if request.is_json:
            return jsonify({'message': message, 'success': False}), 400
        return render_template('auth/signup.html', categories=product_categories, error=message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('auth/login.html', categories=product_categories)
    elif request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else:
            email = request.form.get('email')
            password = request.form.get('password')
        success, message = login_user(email, password)
        if success:
            user = {'email': email}
            session['user'] = user
            cache.delete('homepage')
            flash(message, 'success')
            redirect_url = session.get('redirect_after_login')
            if redirect_url:
                session.pop('redirect_after_login', None)
                return redirect(redirect_url)
            if request.is_json:
                return jsonify({'message': message, 'success': True}), 200
            return redirect(url_for('home'))
        flash(message, 'danger')
        if request.is_json:
            return jsonify({'message': message, 'success': False}), 400
        return render_template('auth/login.html', categories=product_categories, error=message)

@app.route('/logout')
def logout():
    cache.delete('homepage')
    session.pop('user', None)
    session.pop('cart', None)
    session.pop('recent_view', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/product/<product_id>')
def product_detail(product_id):
    product = get_product_details(product_id)
    if product is None:
        flash('Product not found', 'error')
        return redirect(url_for('home'))
    # Update recent views (persisting up to 5 recent product IDs)
    session['recent_view'] = product_id
    if 'recent_views' not in session:
        session['recent_views'] = []
    recent_views = session['recent_views']
    if product_id in recent_views:
        recent_views.remove(product_id)
    recent_views.insert(0, product_id)
    session['recent_views'] = recent_views[:5]
    current_user = session.get('user')
    if current_user:
        user_id = current_user.get('email')
        save_browsing_history(user_id, product_id, product.get('category'))
        update_view_history(user_id, product_id, product.get('category'))
    cache_key = f'similar_products_{product_id}'
    similar_products = cache.get(cache_key)
    if similar_products is None:
        similar_products = []
        current_category = product.get('category', '')
        print(f"DEBUG: Tìm sản phẩm tương tự cho: {product['name']} - Danh mục: {current_category}")
        try:
            current_df = get_dataframe()
            if current_df is not None:
                same_category_products = current_df[current_df['category'] == current_category]
                same_category_products = same_category_products[same_category_products['product_id'] != product_id]
                product_name = product['name'].lower()
                keywords = []
                ignore_words = ['for', 'with', 'and', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'to']
                for word in product_name.split():
                    if len(word) > 3 and word not in ignore_words:
                        keywords.append(word)
                print(f"DEBUG: Các từ khóa chính: {keywords}")
                scored_products = []
                for _, row in same_category_products.iterrows():
                    current_name = row['product_name'].lower()
                    score = 0
                    for keyword in keywords:
                        if keyword in current_name:
                            score += 1
                    try:
                        rating = float(row.get('rating', 0))
                        score += rating / 10
                    except:
                        pass
                    scored_products.append((row, score))
                scored_products.sort(key=lambda x: x[1], reverse=True)
                for row, score in scored_products[:4]:
                    similar_products.append({
                        'product_id': row['product_id'],
                        'name': row['product_name'],
                        'price': row.get('actual_price', 'N/A'),
                        'discounted_price': row.get('discounted_price', 'N/A'),
                        'rating': row.get('rating', 'N/A'),
                        'image_url': row.get('img_link', '/static/img/placeholder.png'),
                        'category': row.get('category', 'Uncategorized'),
                        'similarity_score': score
                    })
        except Exception as e:
            print(f"DEBUG: Lỗi khi tìm sản phẩm tương tự: {e}")
        if len(similar_products) < 4:
            try:
                print(f"DEBUG: Không đủ sản phẩm tương tự, bổ sung bằng sản phẩm phổ biến trong danh mục {current_category}")
                current_df = get_dataframe()
                if current_df is not None:
                    category_products = current_df[current_df['category'] == current_category]
                    category_products['rating_numeric'] = pd.to_numeric(category_products['rating'], errors='coerce')
                    category_products['rating_count_numeric'] = pd.to_numeric(category_products['rating_count'], errors='coerce')
                    category_products['popularity'] = category_products['rating_numeric'] * np.log1p(category_products['rating_count_numeric'])
                    popular_in_category = category_products.sort_values('popularity', ascending=False)
                    existing_ids = [p['product_id'] for p in similar_products] + [product_id]
                    popular_in_category = popular_in_category[~popular_in_category['product_id'].isin(existing_ids)]
                    needed = 4 - len(similar_products)
                    for _, row in popular_in_category.head(needed).iterrows():
                        similar_products.append({
                            'product_id': row['product_id'],
                            'name': row['product_name'],
                            'price': row.get('actual_price', 'N/A'),
                            'discounted_price': row.get('discounted_price', 'N/A'),
                            'rating': row.get('rating', 'N/A'),
                            'image_url': row.get('img_link', '/static/img/placeholder.png'),
                            'category': row.get('category', 'Uncategorized')
                        })
            except Exception as e:
                print(f"DEBUG: Lỗi khi bổ sung sản phẩm phổ biến: {e}")
        cache.set(cache_key, similar_products, timeout=CACHE_TIMEOUT)
    return render_template('product_detail.html',
                           product=product,
                           similar_products=similar_products,
                           categories=product_categories,
                           current_user=current_user)

@app.route('/category/<category>')
def category(category):
    products = get_products_by_category(category)
    current_user = session.get('user')
    return render_template('category.html',
                           category=category,
                           products=products,
                           categories=product_categories,
                           current_user=current_user)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return redirect(url_for('home'))
    current_user = session.get('user')
    if current_user:
        user_id = current_user.get('email')
        if user_id:
            success = save_search_history(user_id, query)
            if not success:
                print(f"DEBUG: Không thể lưu lịch sử tìm kiếm cho query '{query}' của user '{user_id}'")
        else:
            print("DEBUG: Không có user_id trong session.")
    else:
        print("DEBUG: Người dùng chưa đăng nhập, không lưu lịch sử tìm kiếm.")
    results = search_products(query)
    return render_template('search_results.html',
                           query=query,
                           results=results,
                           categories=product_categories,
                           current_user=current_user)

@app.route('/cart')
def cart():
    cart_items = session.get('cart', {})
    cart_products = []
    total = 0
    cache_key = f'cart_{str(cart_items)}'
    cached_cart_data = cache.get(cache_key)
    if cached_cart_data:
        cart_products, total = cached_cart_data
    else:
        for product_id, quantity in cart_items.items():
            product = get_product_details(product_id)
            if product:
                price = product['discounted_price']
                if price == 'N/A':
                    price = product['price']
                try:
                    if isinstance(price, str):
                        price = price.replace('₹', '').replace(',', '').strip()
                    price_num = float(price)
                except:
                    price_num = 0
                item_total = price_num * quantity
                total += item_total
                cart_products.append({
                    'product_id': product_id,
                    'name': product['name'],
                    'price': price,
                    'quantity': quantity,
                    'item_total': item_total,
                    'image_url': product['image_url'],
                    'category': product.get('category', 'Uncategorized')
                })
        cache.set(cache_key, (cart_products, total), timeout=CACHE_TIMEOUT)
    current_user = session.get('user')
    if current_user:
        recommended_products = get_recommended_products_from_file(current_user.get('email'), limit=4)
    else:
        recommended_products = None
    return render_template('cart.html',
                           cart_products=cart_products,
                           total=total,
                           recommended_products=recommended_products,
                           categories=product_categories,
                           current_user=current_user)

@app.route('/add_to_cart/<product_id>', methods=['POST'])
def add_to_cart(product_id):
    current_user = session.get('user')
    if not current_user:
        session['redirect_after_login'] = request.url
        flash('Please login to add products to cart', 'warning')
        return redirect(url_for('require_login'))
    quantity = int(request.form.get('quantity', 1))
    product = get_product_details(product_id)
    if product is None:
        flash('Không tìm thấy sản phẩm', 'error')
        return redirect(url_for('home'))
    cart = session.get('cart', {})
    if product_id in cart:
        cart[product_id] += quantity
    else:
        cart[product_id] = quantity
    session['cart'] = cart
    # Save updated cart categories to disk.
    save_cart_categories(current_user.get('email'), cart)
    clear_cart_cache(cart)
    flash(f"Đã thêm {product['name']} vào giỏ hàng", 'success')
    update_cart_history(current_user.get('email'), product_id)
    next_page = request.form.get('next') or url_for('product_detail', product_id=product_id)
    return redirect(next_page)

@app.route('/require_login')
def require_login():
    if 'redirect_after_login' not in session and request.referrer:
        session['redirect_after_login'] = request.referrer
    previous_url = request.referrer or url_for('home')
    return render_template('require_login.html',
                           categories=product_categories,
                           previous_url=previous_url)

@app.route('/update_cart', methods=['POST'])
def update_cart():
    current_user = session.get('user')
    if not current_user:
        flash('Please login to update cart', 'warning')
        return redirect(url_for('require_login'))
    cart = session.get('cart', {})
    for key, value in request.form.items():
        if key.startswith('quantity_'):
            product_id = key.replace('quantity_', '')
            try:
                quantity = int(value)
                if quantity > 0:
                    cart[product_id] = quantity
                else:
                    if product_id in cart:
                        del cart[product_id]
            except:
                pass
    session['cart'] = cart
    # Save updated cart categories for the user.
    save_cart_categories(current_user.get('email'), cart)
    clear_cart_cache(cart)
    flash('Giỏ hàng đã được cập nhật', 'success')
    return redirect(url_for('cart'))

@app.route('/remove_from_cart/<product_id>')
def remove_from_cart(product_id):
    current_user = session.get('user')
    if not current_user:
        flash('Please login to remove products from cart', 'warning')
        return redirect(url_for('require_login'))
    cart = session.get('cart', {})
    if product_id in cart:
        del cart[product_id]
        session['cart'] = cart
        clear_cart_cache(cart)
        flash('Product has been removed from cart', 'success')
    return redirect(url_for('cart'))

def clear_cart_cache(cart=None):
    try:
        if cart is not None:
            cart_key = f'cart_{str(cart)}'
            cache.delete(cart_key)
        cache.delete('homepage')
        if hasattr(cache, '_cache'):
            cache_keys = list(cache._cache.keys())
            for key in cache_keys:
                if isinstance(key, str) and key.startswith('cart_'):
                    cache.delete(key)
    except Exception as e:
        print(f"DEBUG: Lỗi khi xóa cache giỏ hàng: {e}")

@app.route('/checkout')
def checkout():
    cart_items = session.get('cart', {})
    if not cart_items:
        flash('Your cart is empty', 'info')
        return redirect(url_for('home'))
    cache_key = f'cart_{str(cart_items)}'
    cached_cart_data = cache.get(cache_key)
    if cached_cart_data:
        cart_products, total = cached_cart_data
    else:
        cart_products = []
        total = 0
        for product_id, quantity in cart_items.items():
            product = get_product_details(product_id)
            if product:
                price = product['discounted_price']
                if price == 'N/A':
                    price = product['price']
                try:
                    if isinstance(price, str):
                        price = price.replace('₹', '').replace(',', '').strip()
                    price_num = float(price)
                except:
                    price_num = 0
                item_total = price_num * quantity
                total += item_total
                cart_products.append({
                    'product_id': product_id,
                    'name': product['name'],
                    'price': price,
                    'quantity': quantity,
                    'item_total': item_total,
                    'category': product.get('category', 'Uncategorized')
                })
        cache.set(cache_key, (cart_products, total), timeout=CACHE_TIMEOUT)
    current_user = session.get('user')
    return render_template('checkout.html',
                           cart_products=cart_products,
                           total=total,
                           categories=product_categories,
                           current_user=current_user)

@app.route('/complete_order', methods=['POST'])
def complete_order():
    for key in list(cache.cache.keys()):
        if key.startswith('cart_'):
            cache.delete(key)
    session['cart'] = {}
    flash('Order completed successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/api/preload_data', methods=['GET'])
def preload_data():
    threading.Thread(target=load_model).start()
    threading.Thread(target=load_metadata).start()
    threading.Thread(target=lambda: load_dataframe(None)).start()
    return jsonify({
        'success': True,
        'message': 'Background data loading started'
    })

@app.route('/api/load_status', methods=['GET'])
def load_status():
    return jsonify({
        'model_loaded': model_loaded,
        'metadata_loaded': metadata_loaded,
        'df_loaded': df_loaded,
        'products_count': len(df) if df is not None else 0,
        'categories_count': len(product_categories) if product_categories else 0
    })

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    if request.headers.get('X-Admin-Key') != 'admin-secret-key':
        return jsonify({'error': 'Unauthorized'}), 403
    cache.clear()
    return jsonify({'success': True, 'message': 'Cache cleared'})

def clear_cache_pattern(pattern):
    count = 0
    for key in list(cache.cache.keys()):
        if pattern in key:
            cache.delete(key)
            count += 1
    return count

@app.route('/admin/cache', methods=['GET'])
def admin_cache():
    if request.args.get('key') != 'admin-secret-key':
        return "Unauthorized", 403
    cache_stats = {
        'total_keys': len(cache.cache),
        'product_keys': sum(1 for k in cache.cache.keys() if 'product_' in k),
        'category_keys': sum(1 for k in cache.cache.keys() if 'category_' in k),
        'search_keys': sum(1 for k in cache.cache.keys() if 'search_' in k),
        'cart_keys': sum(1 for k in cache.cache.keys() if 'cart_' in k),
        'recommendation_keys': sum(1 for k in cache.cache.keys() if 'recommendations_' in k),
    }
    return jsonify(cache_stats)

@app.before_request
def before_request():
    request.start_time = datetime.now()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        duration = datetime.now() - request.start_time
        response.headers['X-Request-Duration'] = str(duration.total_seconds())
    return response

def initialize_app():
    global df, product_categories, model, metadata
    global product_encoder, idx_to_product, product_to_name, rating_scaler
    global category_mapping, user_embeddings, product_embeddings
    global model_loaded, metadata_loaded, df_loaded
    print("DEBUG: Initializing application and loading all data...")
    try:
        print("DEBUG: Loading model...")
        model = tf.keras.models.load_model('amazon_recommender_model_updated.keras')
        print("DEBUG: Model loaded successfully")
        model_loaded = True
    except Exception as e:
        print(f"DEBUG: Error loading model: {e}")
        model = None
        model_loaded = False
    try:
        print("DEBUG: Loading metadata...")
        with open('amazon_recommender_metadata_updated.pkl', 'rb') as f:
            metadata = pickle.load(f)
        print("DEBUG: Metadata loaded successfully")
        if metadata is not None:
            product_encoder = metadata.get('product_encoder')
            user_encoder = metadata.get('user_encoder', None)
            idx_to_product = metadata.get('idx_to_product')
            product_to_name = metadata.get('product_to_name')
            rating_scaler = metadata.get('rating_scaler')
            category_mapping = metadata.get('category_mapping')
            user_embeddings = metadata.get('user_embeddings')
            product_embeddings = metadata.get('product_embeddings')
        metadata_loaded = True
    except Exception as e:
        print(f"DEBUG: Error loading metadata: {e}")
        metadata = None
        metadata_loaded = False
    try:
        if os.path.exists('amazon_cleaned.csv'):
            print("DEBUG: Loading full CSV data...")
            df = pd.read_csv('amazon_cleaned.csv')
            print(f"DEBUG: Data loaded; {len(df)} products found")
        else:
            print("DEBUG: CSV data file not found")
            df = None
        if df is not None and 'category' in df.columns:
            product_categories = sorted(df['category'].dropna().unique().tolist())
            print(f"DEBUG: Found {len(product_categories)} product categories")
        else:
            product_categories = []
        df_loaded = True
    except Exception as e:
        print(f"DEBUG: Error loading CSV data: {e}")
        df = None
        df_loaded = False
    if not os.path.exists('static/img'):
        try:
            os.makedirs('static/img')
        except:
            pass
    placeholder_path = 'static/img/placeholder.png'
    if not os.path.exists(placeholder_path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (200, 200), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            d.text((70, 90), "No Image", fill=(150, 150, 150))
            img.save(placeholder_path)
        except:
            print("DEBUG: Could not create placeholder image")
    if products_collection is not None and df is not None:
        try:
            if products_collection.count_documents({}) == 0:
                print("DEBUG: Saving data to MongoDB...")
                process_and_save_data('amazon_cleaned.csv')
                print("DEBUG: Data saved to MongoDB successfully")
        except Exception as e:
            print(f"DEBUG: Error saving data to MongoDB: {e}")
    print("DEBUG: Application initialization complete!")

with app.app_context():
    initialize_app()

@app.context_processor
def inject_preload_script():
    script = """
    <script>
      console.log("DEBUG: Preload script initiated");
      window.addEventListener('load', function() {
          setTimeout(function() {
              console.log("DEBUG: Fetching preload_data endpoint...");
              fetch('/api/preload_data')
                  .then(response => response.json())
                  .then(data => console.log("DEBUG: Preload data response:", data))
                  .catch(error => console.error("DEBUG: Error preloading data:", error));
          }, 2000);
      });
    </script>
    """
    return {'preload_script': script}

if __name__ == '__main__':
    import sys
    debug_mode = "--debug" in sys.argv or DEBUG_MODE
    app.run(debug=debug_mode, threaded=True)
