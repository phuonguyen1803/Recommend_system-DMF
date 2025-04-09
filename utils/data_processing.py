import pandas as pd
import numpy as np
from data.database import save_products
import os

def process_and_save_data(csv_path):
    """
    Đọc dữ liệu từ file CSV, tính popularity và lưu vào MongoDB.
    
    Args:
        csv_path (str): Đường dẫn đến file CSV
    
    Returns:
        pandas.DataFrame hoặc None: DataFrame đã xử lý hoặc None nếu có lỗi
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} không tồn tại.")
        return None
        
    try:
        # Đọc file CSV
        df = pd.read_csv(csv_path)
        print(f"Đã đọc {len(df)} dòng từ {csv_path}.")
        
        # Chuyển đổi kiểu dữ liệu
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0)
        
        # Loại bỏ dữ liệu bị thiếu trong các cột quan trọng
        required_columns = ['product_id', 'product_name']
        df = df.dropna(subset=required_columns)
        
        # Chuẩn hóa category
        if 'category' in df.columns:
            df['category'] = df['category'].str.strip()
        
        # Tính popularity score (rating * log(1 + rating_count))
        df['popularity'] = df['rating'] * np.log1p(df['rating_count'])
        
        # Chuẩn hóa URLs
        if 'img_link' in df.columns:
            # Replace empty URLs with placeholder
            df['img_link'] = df['img_link'].fillna('/static/img/placeholder.png')
            
            # Ensure all URLs are valid
            df['img_link'] = df['img_link'].apply(lambda x: 
                x if isinstance(x, str) and (x.startswith('http') or x.startswith('/')) 
                else '/static/img/placeholder.png')
        
        # Xử lý giá tiền
        for price_column in ['actual_price', 'discounted_price']:
            if price_column in df.columns:
                # Remove currency symbols and commas
                df[price_column] = df[price_column].astype(str)
                df[price_column] = df[price_column].str.replace('₹', '').str.replace(',', '').str.strip()
                
                # Convert to numeric where possible
                df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
                
                # Fill NaN with 'N/A'
                df[price_column] = df[price_column].fillna('N/A')
        
        # Lưu dữ liệu vào MongoDB
        save_products(df)
        
        return df
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        return None

# Xử lý dữ liệu text chuẩn hóa
def preprocess_text(text):
    """
    Chuẩn hóa văn bản (loại bỏ kí tự đặc biệt, chuyển thành lowercase)
    
    Args:
        text (str): Văn bản cần chuẩn hóa
        
    Returns:
        str: Văn bản đã chuẩn hóa
    """
    if not isinstance(text, str):
        return ""
    
    # Chuyển thành lowercase
    text = text.lower()
    
    # Loại bỏ kí tự đặc biệt
    import re
    text = re.sub(r'[^\w\s]', '', text)
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text

# Tạo keywords cho tìm kiếm
def generate_keywords(df, text_columns=['product_name', 'about_product', 'category']):
    """
    Tạo keywords từ các cột văn bản để hỗ trợ tìm kiếm
    
    Args:
        df (pandas.DataFrame): DataFrame cần xử lý
        text_columns (list): Các cột chứa văn bản để tạo keywords
        
    Returns:
        pandas.DataFrame: DataFrame với cột keywords đã được thêm vào
    """
    result_df = df.copy()
    
    # Tạo cột keywords
    result_df['keywords'] = ''
    
    for col in text_columns:
        if col in result_df.columns:
            # Thêm từ của cột vào keywords
            result_df['keywords'] += ' ' + result_df[col].astype(str)
    
    # Chuẩn hóa keywords
    result_df['keywords'] = result_df['keywords'].apply(preprocess_text)
    
    return result_df

if __name__ == "__main__":
    # Chạy hàm xử lý và lưu dữ liệu khi file được chạy trực tiếp
    csv_path = 'amazon_cleaned.csv'
    if os.path.exists(csv_path):
        processed_df = process_and_save_data(csv_path)
        if processed_df is not None:
            # Tạo keywords
            processed_df = generate_keywords(processed_df)
            # Lưu lại file đã xử lý nếu cần
            processed_df.to_csv('amazon_processed.csv', index=False)
    else:
        print(f"File {csv_path} không tồn tại.")