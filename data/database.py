from pymongo import MongoClient
import bcrypt
import time
from urllib.parse import quote_plus
from datetime import datetime

# Kết nối tới MongoDB với retry logic
def connect_mongodb(max_retries=3, retry_delay=2):
    """Connect to MongoDB with retry logic"""
    retries = 0
    # Thay bằng chuỗi kết nối chính xác từ MongoDB Atlas
    username = "nhoktk000"  # Thay bằng username của bạn
    password = "Quan26012004"  # Sử dụng password mới bạn cung cấp
    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)
    uri = f"mongodb+srv://{encoded_username}:{encoded_password}@cluster.ap9ga.mongodb.net/ecommerce_db?retryWrites=true&w=majority"
    # Lưu ý: Lấy chuỗi chính xác từ Atlas (Clusters > Connect > Connect your application) và thay thế uri này nếu cần
    
    while retries < max_retries:
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            print("Kết nối MongoDB thành công!")
            return client
        except Exception as e:
            retries += 1
            print(f"Lỗi kết nối MongoDB (lần thử {retries}/{max_retries}): {str(e)}")
            if retries < max_retries:
                print(f"Thử kết nối lại sau {retry_delay} giây...")
                time.sleep(retry_delay)
            else:
                print(f"Kết nối thất bại sau {max_retries} lần thử.")
    return None

# Hàm lấy IP cục bộ (dùng để debug)
def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Không thể lấy IP"

# Khởi tạo kết nối và collections
try:
    client = connect_mongodb()
    if client:
        db = client['ecommerce_db']
        users_collection = db['users']
        products_collection = db['products']
        browsing_history_collection = db['browsing_history']
        search_history_collection = db['search_history']
        print("Collections initialized successfully.")
    else:
        db = None
        users_collection = None
        products_collection = None
        browsing_history_collection = None
        search_history_collection = None
        print("Không thể khởi tạo collections do lỗi kết nối.")
except Exception as e:
    print(f"Lỗi khởi tạo collections: {str(e)}")
    db = None
    users_collection = None
    products_collection = None
    browsing_history_collection = None
    search_history_collection = None

# Hàm lưu thông tin người dùng khi đăng ký
def signup_user(username, email, password):
    """
    Đăng ký người dùng mới
    
    Args:
        username (str): Tên người dùng
        email (str): Email (dùng làm username đăng nhập)
        password (str): Mật khẩu
        
    Returns:
        tuple: (success, message)
    """
    if users_collection is None:
        return False, "Không thể kết nối tới cơ sở dữ liệu."
        
    if users_collection.find_one({'email': email}):
        return False, "Email đã tồn tại."
    
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': time.time(),
            'cart_history': [],
            'view_history': []
        }
        users_collection.insert_one(user)
        return True, "Đăng ký thành công."
    except Exception as e:
        return False, f"Lỗi khi đăng ký: {str(e)}"

# Hàm kiểm tra đăng nhập
def login_user(email, password):
    """
    Kiểm tra thông tin đăng nhập
    
    Args:
        email (str): Email người dùng
        password (str): Mật khẩu
        
    Returns:
        tuple: (success, message)
    """
    if users_collection is None:
        return False, "Không thể kết nối tới cơ sở dữ liệu."
        
    try:
        user = users_collection.find_one({'email': email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return True, "Đăng nhập thành công."
        return False, "Email hoặc mật khẩu không đúng."
    except Exception as e:
        return False, f"Lỗi khi đăng nhập: {str(e)}"

# Hàm lưu dữ liệu sản phẩm từ DataFrame vào MongoDB
def save_products(df):
    """
    Lưu dữ liệu sản phẩm từ DataFrame vào MongoDB
    
    Args:
        df (pandas.DataFrame): DataFrame chứa dữ liệu sản phẩm
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    if products_collection is None:
        print("Không thể kết nối tới cơ sở dữ liệu.")
        return False
        
    try:
        if products_collection.count_documents({}) > 0:
            print("Collection sản phẩm đã có dữ liệu. Bỏ qua.")
            return True
            
        records = df.to_dict('records')
        for record in records:
            if 'product_id' in record and 'user_id' in record:
                record['_id'] = f"{record['product_id']}_{record['user_id']}"
            else:
                record['_id'] = record.get('product_id', str(time.time()))
        
        products_collection.insert_many(records, ordered=False)
        print(f"Đã lưu {len(records)} sản phẩm vào MongoDB.")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu sản phẩm: {str(e)}")
        return False

# Hàm lưu lịch sử xem sản phẩm
def save_browsing_history(user_id, product_id, product_category=None):
    """
    Lưu lịch sử xem sản phẩm của người dùng
    
    Args:
        user_id (str): ID người dùng (email)
        product_id (str): ID sản phẩm đã xem
        product_category (str): Danh mục sản phẩm
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    if browsing_history_collection is None:
        return False
        
    try:
        history_record = {
            'user_id': user_id,
            'product_id': product_id,
            'timestamp': time.time()
        }
        if product_category:
            history_record['category'] = product_category
            
        browsing_history_collection.insert_one(history_record)
        return True
    except Exception as e:
        print(f"Lỗi khi lưu lịch sử xem: {str(e)}")
        return False

# Hàm lưu lịch sử tìm kiếm với thời gian vào collection 'search_history'
def save_search_history(user_id, search_query):
    """
    Lưu lịch sử tìm kiếm của người dùng vào MongoDB collection 'search_history'
    
    Args:
        user_id (str): ID người dùng (email)
        search_query (str): Từ khóa tìm kiếm
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    if search_history_collection is None:
        print("Lỗi: search_history_collection không được khởi tạo. Kiểm tra kết nối MongoDB.")
        return False
        
    try:
        current_time = datetime.now()
        search_record = {
            'user_id': user_id,
            'query': search_query,
            'timestamp': time.time(),
            'search_time': current_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        search_history_collection.insert_one(search_record)
        print(f"Thành công: Đã lưu lịch sử tìm kiếm '{search_query}' cho user '{user_id}' vào 'search_history'")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu lịch sử tìm kiếm cho user '{user_id}', query '{search_query}': {str(e)}")
        return False

# Hàm lấy lịch sử xem gần đây
def get_recent_views(user_id, limit=5):
    """
    Lấy danh sách sản phẩm người dùng đã xem gần đây
    
    Args:
        user_id (str): ID người dùng (email)
        limit (int): Số lượng tối đa kết quả
        
    Returns:
        list: Danh sách ID sản phẩm đã xem gần đây
    """
    if browsing_history_collection is None:
        return []
        
    try:
        history = list(browsing_history_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(limit))
        product_ids = [item['product_id'] for item in history]
        return product_ids
    except Exception as e:
        print(f"Lỗi khi lấy lịch sử xem: {str(e)}")
        return []

# Hàm lấy danh mục xem gần đây
def get_recent_categories(user_id, limit=3):
    """
    Lấy danh sách danh mục người dùng đã xem gần đây
    
    Args:
        user_id (str): ID người dùng (email)
        limit (int): Số lượng tối đa danh mục
        
    Returns:
        list: Danh sách danh mục đã xem gần đây
    """
    if browsing_history_collection is None:
        return []
        
    try:
        history = list(browsing_history_collection.find(
            {'user_id': user_id, 'category': {'$exists': True}}
        ).sort('timestamp', -1).limit(20))
        
        categories = []
        for item in history:
            category = item.get('category')
            if category and category not in categories:
                categories.append(category)
                if len(categories) >= limit:
                    break
        return categories
    except Exception as e:
        print(f"Lỗi khi lấy danh mục gần đây: {str(e)}")
        return []

# Hàm cập nhật lịch sử giỏ hàng
def update_cart_history(user_email, product_id):
    if users_collection is None:
        return False
    try:
        users_collection.update_one(
            {'email': user_email},
            {'$push': {'cart_history': {'product_id': product_id, 'timestamp': time.time()}}},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Lỗi khi cập nhật lịch sử giỏ hàng: {str(e)}")
        return False

# Hàm cập nhật lịch sử xem sản phẩm
def update_view_history(user_email, product_id, product_category=None):
    if users_collection is None:
        return False
    try:
        users_collection.update_one(
            {'email': user_email},
            {'$push': {'view_history': {'product_id': product_id, 'category': product_category, 'timestamp': time.time()}}},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Lỗi khi cập nhật lịch sử xem: {str(e)}")
        return False