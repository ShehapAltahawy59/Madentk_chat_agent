from typing import List, Optional, Literal
import os
import json
import numpy as np
import unicodedata
import random
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

# Load environment variables
load_dotenv()

# Import ML dependencies directly (Full ML setup) - but delay heavy initialization
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from fuzzywuzzy import process
from algoliasearch.search.client import SearchClientSync as SearchClient
import asyncio


print("✅ ML dependencies imported successfully (models will load at startup)")

# Active user context
active_user_id: Optional[str] = None
active_where: Optional[str] = None

def set_active_user_id(user_id: Optional[str]) -> None:
    global active_user_id
    active_user_id = user_id

def get_active_user_id() -> Optional[str]:
    return active_user_id

def set_active_where(where_value: Optional[str]) -> None:
    global active_where
    active_where = where_value

def get_active_where() -> Optional[str]:
    return active_where

# Initialize Firestore
if not firebase_admin._apps:
    cred_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "foodorderapp.json")
    
    # Debug: Print what we received (first 100 chars)
    print(f"🔍 Received GOOGLE_APPLICATION_CREDENTIALS length: {len(cred_env) if cred_env else 0}")
    print(f"🔍 First 100 chars: {cred_env[:100] if cred_env else 'None'}")
    print(f"🔍 Last 100 chars: {cred_env[-100:] if cred_env and len(cred_env) > 100 else 'None'}")
    
    # Accept file path, base64 encoded JSON, or raw JSON string
    if os.path.isfile(cred_env):
        cred_obj = credentials.Certificate(cred_env)
        print("✅ Using Firebase credentials from file")
    else:
        try:
            # First try to decode as base64 (recommended for secrets)
            import base64
            decoded_cred = base64.b64decode(cred_env).decode('utf-8')
            cred_info = json.loads(decoded_cred)
            cred_obj = credentials.Certificate(cred_info)
            print("✅ Using Firebase credentials from base64 encoded JSON")
        except Exception as e:
            print(f"⚠ Base64 decode failed: {e}")
            try:
                # Fallback to direct JSON parsing
                cred_info = json.loads(cred_env)
                cred_obj = credentials.Certificate(cred_info)
                print("✅ Using Firebase credentials from raw JSON string")
            except json.JSONDecodeError as json_e:
                print(f"⚠ JSON parse failed: {json_e}")
                print(f"⚠ Raw cred_env content: '{cred_env}'")
                raise ValueError(
                    f"GOOGLE_APPLICATION_CREDENTIALS must be a valid file path, base64 encoded JSON, or raw JSON string. Base64 error: {e}, JSON error: {json_e}"
                )
    firebase_admin.initialize_app(cred_obj)

# Login to Hugging Face Hub if token provided
_hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if _hf_token:
    try:
        hf_login(token=_hf_token)
    except Exception as _e:
        print(f"⚠ Hugging Face login failed: {_e}")

db = firestore.client()

# --- Algolia Setup (v4) ---
ALGOLIA_APP_ID = os.environ.get("ALGOLIA_APP_ID")
ALGOLIA_API_KEY = os.environ.get("ALGOLIA_API_KEY")
ALGOLIA_CATEGORIES_INDEX = os.environ.get("ALGOLIA_CATEGORIES_INDEX", "categories_index")
ALGOLIA_ITEMS_INDEX = os.environ.get("ALGOLIA_ITEMS_INDEX", "items_index")

algolia_client = None

if ALGOLIA_APP_ID and ALGOLIA_API_KEY:
    try:
        algolia_client = SearchClient(ALGOLIA_APP_ID, ALGOLIA_API_KEY)
        print("✅ Algolia client v4 initialized")
    except Exception as e:
        print(f"⚠ Failed to initialize Algolia: {e}")
else:
    print("⚠ Missing ALGOLIA_APP_ID or ALGOLIA_API_KEY; Algolia search disabled")

def _algolia_search(index_name: str, query: str, params: dict) -> dict:
    """Algolia v4 batch search. Returns per-index result dict with 'hits'."""
    if not algolia_client:
        return {"hits": []}
    try:
        search_params = {
            "search_method_params": {
                "requests": [
                    {
                        "indexName": index_name,
                        "query": query,
                        **(params or {}),
                    }
                ]
            }
        }
        resp = algolia_client.search(**search_params)
        # Convert to dict-like
        if hasattr(resp, 'to_dict'):
            data = resp.to_dict()
        elif hasattr(resp, 'to_json'):
            import json as _json
            data = _json.loads(resp.to_json())
        else:
            data = resp
        results = (data or {}).get("results") or (data or {}).get("responses")
        if results and isinstance(results, list):
            return results[0]
        return {"hits": []}
    except Exception as e:
        print(f"⚠ Algolia search call failed: {e}")
        return {"hits": []}

#--- Arabic Normalization ---
def normalize_arabic(text: str) -> str:
    """
    Enhanced Arabic text normalization for better fuzzy matching.
    Handles various Arabic character variations and diacritics.
    """
    if not text:
        return text
    
    # Basic cleaning
    text = text.strip().lower()
    
    # Remove diacritics (tashkeel) first
    diacritics = 'ًٌٍَُِّْٰ'
    for diacritic in diacritics:
        text = text.replace(diacritic, '')
    
    # Comprehensive character normalization
    replacements = {
        # Alef variations
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
        # Yeh variations
        'ي': 'ى', 'ئ': 'ى', 'ء': 'ى',
        # Teh marbuta
        'ة': 'ه',
        # Hamza variations
        'ؤ': 'و', '۽': 'و',
        # Other common variations
        'ك': 'ك', 'ڪ': 'ك',  # Kaf variations
        'گ': 'ك',  # Persian Kaf
        'ڤ': 'ف',  # Veh
        'چ': 'ج',  # Cheh
        'پ': 'ب',  # Peh
        'ژ': 'ز',  # Zheh
        'ڨ': 'ق',  # Qaf with three dots
        'ڧ': 'ق',  # Qaf with dot above
        'ڢ': 'ف',  # Feh with dot below
        'ڡ': 'ف',  # Feh with dot moved below
        'ڦ': 'ف',  # Feh with three dots below
        'ڥ': 'ف',  # Feh with three dots pointing down
        'ڨ': 'ق',  # Qaf with three dots above
        'ڧ': 'ق',  # Qaf with dot above
        'ڢ': 'ف',  # Feh with dot below
        'ڡ': 'ف',  # Feh with dot moved below
        'ڦ': 'ف',  # Feh with three dots below
        'ڥ': 'ف',  # Feh with three dots pointing down
        # Remove common punctuation and spaces
        '،': '', '؛': '', '؟': '', '!': '', 'ـ': '', 'ـ': '',
        # Normalize spaces
        '\u200f': ' ', '\u200e': ' ', '\u200d': '',  # RTL/LTR marks
        '\u00a0': ' ',  # Non-breaking space
        '\u2000': ' ', '\u2001': ' ', '\u2002': ' ', '\u2003': ' ',
        '\u2004': ' ', '\u2005': ' ', '\u2006': ' ', '\u2007': ' ',
        '\u2008': ' ', '\u2009': ' ', '\u200a': ' ', '\u200b': '',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra spaces and clean up
    text = ' '.join(text.split())
    
    return text

def advanced_arabic_fuzzy_match(query: str, choices: list, threshold: int = 70) -> list:
    """
    Advanced fuzzy matching for Arabic text with multiple strategies.
    """
    if not query or not choices:
        return []
    
    # Normalize the query
    normalized_query = normalize_arabic(query)
    
    results = []
    
    for choice in choices:
        if isinstance(choice, dict):
            # Handle dictionary choices (like item data)
            name_ar = normalize_arabic(choice.get("name_ar", ""))
            name_en = normalize_arabic(choice.get("name_en", ""))
            choice_text = f"{name_ar} {name_en}".strip()
        else:
            # Handle string choices
            choice_text = normalize_arabic(str(choice))
        
        if not choice_text:
            continue
            
        # Multiple matching strategies
        scores = []
        
        # 1. Direct fuzzy match
        direct_score = process.extractOne(normalized_query, [choice_text])[1]
        scores.append(direct_score)
        
        # 2. Partial match (for compound words)
        words_query = normalized_query.split()
        words_choice = choice_text.split()
        
        if len(words_query) > 1:
            # Check if all query words are in choice
            word_matches = 0
            for word in words_query:
                if any(process.extractOne(word, [w])[1] >= 80 for w in words_choice):
                    word_matches += 1
            partial_score = (word_matches / len(words_query)) * 100
            scores.append(partial_score)
        
        # 3. Substring match (for partial matches)
        if normalized_query in choice_text or choice_text in normalized_query:
            scores.append(90)
        
        # 4. Character-based similarity (for typos)
        char_similarity = process.extractOne(normalized_query, [choice_text])[1]
        scores.append(char_similarity)
        
        # Take the best score
        best_score = max(scores)
        
        if best_score >= threshold:
            result = choice.copy() if isinstance(choice, dict) else choice
            if isinstance(result, dict):
                result['match_score'] = best_score
            results.append((result, best_score))
    
    # Sort by score and return
    results.sort(key=lambda x: x[1], reverse=True)
    return [result[0] for result in results]

# --- Time-based Meal Keywords Tool ---
def suggest_meal_keywords(now: Optional[datetime] = None) -> List[str]:
    """Return Arabic keywords for the current meal time in Egypt.
    Morning (5-11): إفطار/فطار; Noon (11-16): غداء; Evening (16-22): عشاء; Late (22-2): سناكس.
    """
    current_time = now or datetime.now()
    hour = current_time.hour
    if 5 <= hour < 11:
        return [
            "فطار", "سندوتش", "جبنه", "بيض", "فول", "فلافل", "طعمية", "بطاطس",
            "كروسانت", "توست", "لبنة", "زعتر", "شاي", "قهوة"
        ]
    if 11 <= hour < 16:
        return [
            "غداء", "وجبة", "رز", "فراخ", "لحمة", "شاورما", "كشري", "برجر",
            "بيتزا", "مكرونة", "وجبة اقتصادية", "بطاطس"
        ]
    if 16 <= hour < 22:
        return [
            "عشاء", "سندوتش", "شاورما", "برجر", "كبدة", "سجق", "طاسة", "فتة",
            "كريب", "بطاطس", "تشكن", "تيكا"
        ]
    # Late night
    return [
        "سناكس", "سناك", "سندوتش", "بطاطس", "نوجتس", "كريب", "وافل", "كريب حلو",
        "كريب حادق", "برجر"
    ]

# --- Firestore Tools ---
REQUIRED_FIELDS = {
    "addressid": "",
    "cancelreason": "",
    "delivery": {"name": "", "phone": ""},
    "deliveryCost": "",
    "itemsinorder": [],
    "lat": "",
    "long": "",
    "notes": "",
    "orderid": None,
    "phoneid": "",
    "picked": False,
    "resturant": "",
    "status": "1",
    "time": None,
    "totalprice": "",
    "user_id": "",
    "username": "",
    "where": ""
}

def insert_order(order_data: dict) -> dict:
    """
    Places a new order into Firestore.
    In particular, "itemsinorder" is a list of strings, each should have
    4 colons exactly (5 fields), even if some fields are empty.
    Args:
        order_data (dict): Must match this structure:
            {
                "itemsize":str,# empty if missing
                "addons":str,# empty if missing
                "addressid": str, #addrees of user where order will be delivered
                "cancelreason": "",  # constant
                "delivery": {"name": "", "phone": ""},  # constant
                "deliveryCost":int, 
                "itemsinorder": [ "itemid:item count:{itemsize}:{addons}:item price", ... ],
                "lat": "", # constant
                "long": "", # constant
                "notes": str,
                "orderid": int,  # random if missing
                "phoneid": str,
                "picked": False,  # constant
                "resturant": str,  # restaurant id
                "status": "1",  # constant
                "time": datetime,  # current if missing
                "totalprice": str, # total amount including delivery cost
                "user_id": str,
                "username": str,
                "discountAmount" : 0 # constant
                "updatedAt": datetime,  # current if missing
                "where": "quweisna"  # constant
            }
        If any value is unknown, set it to "" or the constant default above.
        Never omit keys — always include all.

    Returns:
        dict: Confirmation with order_id and summary.
    """
    try:
        for key, default in REQUIRED_FIELDS.items():
            order_data.setdefault(key, default)
        if not order_data["orderid"]:
            order_data["orderid"] = random.randint(1000000000, 9999999999)
        if not order_data["time"]:
            order_data["time"] = datetime.now()
        ctx_where = get_active_where()
        if ctx_where:
            order_data["where"] = ctx_where
        # Visibility when LLM triggers order placement
        print(
            f"🧾 insert_order called | orderid={order_data.get('orderid')} "
            f"user_id={order_data.get('user_id')} resturant={order_data.get('resturant')} "
            f"total={order_data.get('totalprice')} items={len(order_data.get('itemsinorder', []))} where={order_data.get('where')}"
        )
        db.collection("orders").add(order_data)
        return {"status": "success", "message": f"✅ Order {order_data['orderid']} placed successfully."}
    except Exception as e:
        print(f"⚠ Error in insert_order: {e}")
        return {"status": "error", "message": str(e)}

def get_user_by_id(user_id: str) -> dict:
    try:
        query = db.collection("users").where("user_id", "==", user_id).limit(1).stream()
        user_doc = next(query, None)
        if user_doc:
            print("✅ Found user by user_id")
            return user_doc.to_dict()
        else:
            print("⚠ No user found with that user_id")
            return None
    except Exception as e:
        print(f"⚠ Error fetching user: {e}")
        return None

def get_restaurant_by_id(restaurant_id: str) -> dict:
    try:
        ctx_where = get_active_where() or "quweisna"
        print(f"🔍 Getting restaurant {restaurant_id} in location: {ctx_where}")
        
        doc = db.collection("categories").document(restaurant_id).get()
        if doc.exists:
            restaurant_data = doc.to_dict()
            # Check if the restaurant is in the correct location
            if restaurant_data.get("where") == ctx_where:
                return restaurant_data
            else:
                print(f"⚠ Restaurant {restaurant_id} not found in location {ctx_where}")
                return None
        return None
    except Exception as e:
        print(f"⚠ Error fetching restaurant: {e}")
        return None

def get_item_by_id(item_id: str, restaurant_id: Optional[str] = None) -> dict:
    try:
        ctx_where = get_active_where() or "quweisna"
        print(f"🔍 Getting item {item_id} in location: {ctx_where}")
        
        query = db.collection("items").where("item_id", "==", item_id)
        if restaurant_id:
            query = query.where("item_cat", "==", restaurant_id)
        
        doc = next(query.stream(), None)
        if doc:
            item_data = doc.to_dict()
            item_restaurant_id = item_data.get("item_cat", "")
            
            # Check if the item's restaurant is in the current location
            restaurant_doc = db.collection("categories").document(item_restaurant_id).get()
            if restaurant_doc.exists:
                restaurant_data = restaurant_doc.to_dict()
                if restaurant_data.get("where") == ctx_where:
                    return item_data
                else:
                    print(f"⚠ Item {item_id} not in location {ctx_where}")
                    return None
            else:
                print(f"⚠ Restaurant {item_restaurant_id} not found for item {item_id}")
                return None
        return None
    except Exception as e:
        print(f"⚠ Error fetching item: {e}")
        return None

def get_items_in_restaurant(restaurant_id: str) -> List[dict]:
    try:
        ctx_where = get_active_where() or "quweisna"
        print(f"🔍 Getting items for restaurant {restaurant_id} in location: {ctx_where}")
        
        # Check if restaurant is in the current location
        restaurant_doc = db.collection("categories").document(restaurant_id).get()
        if not restaurant_doc.exists:
            print(f"⚠ Restaurant {restaurant_id} not found")
            return []
        
        restaurant_data = restaurant_doc.to_dict()
        if restaurant_data.get("where") != ctx_where:
            print(f"⚠ Restaurant {restaurant_id} not in location {ctx_where}")
            return []
        
        query = db.collection("items").where("item_cat", "==", restaurant_id).stream()
        return [doc.to_dict() for doc in query]
    except Exception as e:
        print(f"⚠ Error fetching items: {e}")
        return []

def search_restaurant_by_name(name: str) -> List[dict]:
    """
    Search restaurants using Algolia `categories_index` with location filter.
    """
    ctx_where = get_active_where()
    if not algolia_client:
        print("⚠ Algolia client unavailable; returning empty result")
        return []
    try:
        params = {
            "filters": f"where:'{ctx_where}'",
            "hitsPerPage": 10,
        }
        res = _algolia_search(ALGOLIA_CATEGORIES_INDEX, name , params)
        hits = res.get("hits", []) if isinstance(res, dict) else []
        print(f"Algolia restaurant search for '{name}' in {ctx_where}: {len(hits)} hits")
        return hits
    except Exception as e:
        print(f"⚠ Algolia restaurant search error: {e}")
        return []

def get_item_by_name(item_name: str, restaurant_id: Optional[str] = None) -> List[dict]:
    """
    Search items using Algolia `items_index` with optional restaurant and location filters.
    """
    ctx_where = get_active_where() or "quweisna"
    if not algolia_client:
        print("⚠ Algolia client unavailable; returning empty result")
        return []
    try:
        if restaurant_id:
            params = {
                "filters": f"item_cat:'{restaurant_id}'",
                    "hitsPerPage": 30,
                }
        else:
                params = {
                    "hitsPerPage": 30,
                }
        
        res = _algolia_search(ALGOLIA_ITEMS_INDEX, item_name , params)
        hits = res.get("hits", []) if isinstance(res, dict) else []
        print(f"Algolia item search for '{item_name}' : {len(hits)} hits")
        print(hits[0])
        return hits
    except Exception as e:
        print(f"⚠ Algolia item search error: {e}")
        return []

def get_delivery_cost(area_name: str) -> Optional[float]:
    """
    Return delivery cost for a given area from Firestore.
    - Reads collection "regions", document "elmnofia"
    - Looks into array field delivery_zones[0]
    - Finds object where zone_name == area_name
    - Returns zone_cost as float if found, else None
    """
    try:
        print(f"🔍 Fetching delivery cost for area: {area_name}")
        doc_ref = db.collection("regions").document("elmnofia").get()
        if not doc_ref.exists:
            print("⚠ regions/elmnofia document not found")
            return None
        data = doc_ref.to_dict() or {}
        city_data = data.get("cities")
        city = city_data[0]
        delivery_zones = city.get("delivery_zones") or []
        if not isinstance(delivery_zones, list) or not delivery_zones:
            print("⚠ delivery_zones is missing or not a list")
            return None
        first_zone_group = delivery_zones[0]
        # The data shape could be either a list of zones or an object with a zones array
        zones = None
        if isinstance(first_zone_group, list):
            zones = first_zone_group
        elif isinstance(first_zone_group, dict):
            # Common patterns: { zones: [...] } or direct zone objects inside array
            zones = first_zone_group.get("zones") if isinstance(first_zone_group.get("zones"), list) else delivery_zones
        else:
            print("⚠ Unrecognized delivery_zones[0] structure")
            return None
        target_normalized = (area_name or "").strip().lower()
        for z in zones:
            if not isinstance(z, dict):
                continue
            name = str(z.get("zone_name", "")).strip().lower()
            if name == target_normalized:
                cost = z.get("zone_cost")
                try:
                    return float(cost)
                except Exception:
                    print(f"⚠ zone_cost not numeric: {cost}")
                    return None
        print(f"⚠ No matching zone_name found for '{area_name}'")
        return None
    except Exception as e:
        print(f"⚠ Error fetching delivery cost: {e}")
        return None
#--- Vectorstore Setup with Startup Loading ---
device = 'cpu'
embedding_model = None
vectorstore = None
reranker_model = None

def get_embedding_model():
    """Get embedding model (loaded at startup)"""
    global embedding_model
    return embedding_model

def get_vectorstore():
    """Get vectorstore (loaded at startup)"""
    global vectorstore
    return vectorstore

def get_reranker():
    """Get reranker model (loaded at startup)"""
    global reranker_model
    return reranker_model

def create_vector_database():
    """
    Create and populate the vector database with documents from Firebase.
    This function fetches restaurants and items from Firestore and creates
    embeddings for semantic search.
    """
    try:
        print("🔄 Creating vector database...")
        
        # Get embedding model
        embedding_model = get_embedding_model()
        if not embedding_model:
            print("⚠ Embedding model not available")
            return False
        
        # Prepare documents
        documents = []
        
        # Add restaurant documents
        print("🔄 Fetching restaurants...")
        restaurants = []
        for doc in db.collection("categories").stream():
            restaurants.append(doc.to_dict())
        
        for restaurant in restaurants:
            # Create restaurant document
            content_parts = []
            if restaurant.get("name_en"):
                content_parts.append(f"Restaurant: {restaurant['name_en']}")
            if restaurant.get("name_ar"):
                content_parts.append(f"مطعم: {restaurant['name_ar']}")
            if restaurant.get("description_en"):
                content_parts.append(f"Description: {restaurant['description_en']}")
            if restaurant.get("description_ar"):
                content_parts.append(f"الوصف: {restaurant['description_ar']}")
            
            if content_parts:
                doc = Document(
                    page_content=" | ".join(content_parts),
                    metadata={
                        "type": "restaurant",
                        "restaurant_id": restaurant.get("id", ""),
                        "name_en": restaurant.get("name_en", ""),
                        "name_ar": restaurant.get("name_ar", ""),
                        "description_en": restaurant.get("description_en", ""),
                        "description_ar": restaurant.get("description_ar", ""),
                        "where": restaurant.get("where", "")  # Use actual where from data
                    }
                )
                documents.append(doc)
        
        # Add item documents
        print("🔄 Fetching items...")
        items = []
        for doc in db.collection("items").stream():
            items.append(doc.to_dict())
        
        # Get restaurant where mapping
        restaurant_where_map = {}
        for doc in db.collection("categories").stream():
            restaurant_data = doc.to_dict()
            restaurant_where_map[doc.id] = restaurant_data.get("where", "")
        
        for item in items:
            # Get restaurant's where value
            restaurant_id = item.get("item_cat", "")
            item_where = restaurant_where_map.get(restaurant_id, "")
            
            # Create item document
            content_parts = []
            if item.get("name_en"):
                content_parts.append(f"Item: {item['name_en']}")
            if item.get("name_ar"):
                content_parts.append(f"الوجبة: {item['name_ar']}")
            if item.get("description_en"):
                content_parts.append(f"Description: {item['description_en']}")
            if item.get("description_ar"):
                content_parts.append(f"الوصف: {item['description_ar']}")
            if item.get("price"):
                content_parts.append(f"Price: {item['price']}")
            
            if content_parts:
                doc = Document(
                    page_content=" | ".join(content_parts),
                    metadata={
                        "type": "item",
                        "item_id": item.get("item_id", ""),
                        "restaurant_id": item.get("item_cat", ""),
                        "name_en": item.get("name_en", ""),
                        "name_ar": item.get("name_ar", ""),
                        "description_en": item.get("description_en", ""),
                        "description_ar": item.get("description_ar", ""),
                        "price": item.get("price", ""),
                        "where": item_where  # Use restaurant's where value
                    }
                )
                documents.append(doc)
        
        print(f"🔄 Created {len(documents)} documents from all locations")
        
        # Create vectorstore
        chroma_path = os.environ.get("CHROMA_DB_DIR", "chroma_db")
        collection_name = "food_data"
        
        # Remove existing directory if it exists
        import shutil
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            print(f"🔄 Removed existing Chroma DB at {chroma_path}")
        
        # Create new vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=chroma_path
        )
        
        # Persist the database
        vectorstore.persist()
        
        print(f"✅ Vector database created successfully with {len(documents)} documents from all locations")
        return True
        
    except Exception as e:
        print(f"⚠ Error creating vector database: {e}")
        return False


# Initialize models at startup
print("🔄 Initializing ML models at startup...")

try:
    print("🔄 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={
            'device': device
        }
    )
    print("✅ Embedding model initialized")
    
    # Load vectorstore
    chroma_path = os.environ.get("CHROMA_DB_DIR", "chroma_db")
    collection_name = "food_data"
    
    print(f"🔍 Looking for Chroma DB at: {chroma_path}")
    print(f"🔍 Path exists: {os.path.exists(chroma_path)}")
    print(f"🔍 Is directory: {os.path.isdir(chroma_path) if os.path.exists(chroma_path) else False}")
    
    # Check if directory exists and contains valid Chroma files
    chroma_valid = False
    if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
        # Check for Chroma database files
        chroma_files = [f for f in os.listdir(chroma_path) if f.endswith('.sqlite3') or f.endswith('.parquet')]
        if chroma_files:
            try:
                print("🔄 Loading vectorstore...")
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=embedding_model,
                    persist_directory=chroma_path
                )
                # Test if vectorstore actually has data
                test_docs = vectorstore.similarity_search("test", k=1)
                print("✅ Vectorstore loaded from existing directory")
                chroma_valid = True
            except Exception as e:
                print(f"⚠ Error loading vectorstore: {e}")
                vectorstore = None
        else:
            print(f"⚠ Chroma DB directory exists but contains no valid database files")
    
    if not chroma_valid:
        print(f"⚠ Chroma DB directory not found or invalid at {chroma_path}")
        # Create the vector database
        if create_vector_database():
            try:
                print("🔄 Loading newly created vectorstore...")
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=embedding_model,
                    persist_directory=chroma_path
                )
                print("✅ Vectorstore loaded from newly created directory")
            except Exception as e:
                print(f"⚠ Error loading newly created vectorstore: {e}")
                vectorstore = None
        else:
            print("⚠ Failed to create vector database")
            vectorstore = None
            # List contents of current directory to debug
            try:
                print(f"🔍 Current directory contents: {os.listdir('.')}")
            except Exception as e:
                print(f"⚠ Could not list directory: {e}")
    
    # Load CrossEncoder reranker
    try:
        print("🔄 Loading CrossEncoder reranker...")
        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", 
            device=device,
            max_length=512  # Limit sequence length to save memory
        )
        print("✅ CrossEncoder reranker initialized")
    except Exception as e:
        print(f"⚠ Failed to initialize CrossEncoder reranker: {e}")
        reranker_model = None
    
    print("✅ ML models initialization completed")
    
except Exception as e:
    print(f"⚠ Failed to initialize ML models: {e}")
    embedding_model = None
    vectorstore = None

#Semantic Search Tool
def search_semantic(query: str, scope: Optional[Literal["item", "restaurant"]] = None, k: int = 20) -> List[dict]:
    vs = get_vectorstore()
    if vs is None:
        print("⚠ Vectorstore not available for semantic search")
        return []
    
    try:
        normalized_query = normalize_arabic(query)
        ctx_where = get_active_where() or "quweisna"
        
        print(f"🔍 Active where context: {ctx_where}")
        
        # First retrieval: by location
        where_filters = {"where": {"$eq": ctx_where}}
        print(f"🔍 First retrieval with where filter: {where_filters}")
        docs_local = vs.max_marginal_relevance_search(normalized_query, k=k*2, fetch_k=80, filter=where_filters)
        
        # Second filtering: by scope if specified
        if scope and docs_local:
            print(f"🔍 Second filtering by scope: {scope}")
            filtered_docs = []
            for doc in docs_local:
                if doc.metadata.get("type") == scope:
                    filtered_docs.append(doc)
            docs_local = filtered_docs[:k]  # Limit to requested k
            print(f"🔍 After scope filtering: {len(docs_local)} results")
        if docs_local:
            reranker = get_reranker()
            if reranker:
                try:
                    print("🔄 Running reranking...")
                    # Process in smaller batches to reduce memory usage
                    batch_size = 8
                    all_scores = []
                    for i in range(0, len(docs_local), batch_size):
                        batch = docs_local[i:i + batch_size]
                        pairs = [(normalized_query, doc.page_content) for doc in batch]
                        batch_scores = reranker.predict(pairs)
                        all_scores.extend(batch_scores)
                    
                    # Rerank based on scores
                    reranked = [docs_local[i] for i in np.argsort(all_scores)[::-1]]
                    docs_local = reranked[:k]
                    print("✅ Reranking completed")
                except Exception as e:
                    print(f"⚠ Reranking failed, using original results: {e}")
                    docs_local = docs_local[:k]
            else:
                print("⚠ Reranker not available, using original results")
                docs_local = docs_local[:k]
        print(f"Semantic search for '{query}' (normalized: '{normalized_query}') returned {len(docs_local)} results")
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs_local]
    except Exception as e:
        print(f"⚠ Vector search error: {e}")
        return []

# --- Load restaurant_data and name mapping (preserved) ---

# Composed recommendation tool
def recommend_time_based_suggestions(budget_egp: Optional[float] = None, k: int = 20) -> List[dict]:
    """Use time-based keywords with semantic search and Algolia item search.
    Optional budget filters by price metadata.
    """
    try:
        keywords = suggest_meal_keywords()
        print(f"🔍 Using meal keywords: {keywords}")
        results: List[dict] = []
        seen = set()
        for kw in keywords:
            # 1) Semantic search
            sem_hits = search_semantic(kw, scope=None, k=max(2, k // 2)) 
            # 2) Algolia item search
            alg_hits = get_item_by_name(kw) 

            # Normalize Algolia hits to same shape
            normalized_alg_hits: List[dict] = []
            for ah in alg_hits:
                meta = {
                    "type": "item",
                    "item_id": ah.get("item_id") or ah.get("id") or "",
                    "restaurant_id": ah.get("item_cat", ""),
                    "name_en": ah.get("name_en", ""),
                    "name_ar": ah.get("name_ar", ""),
                    "description_en": ah.get("description_en", ""),
                    "description_ar": ah.get("description_ar", ""),
                    "price": ah.get("price", ""),
                    "where": ah.get("where", get_active_where() or "")
                }
                content_parts = []
                if meta["name_en"]:
                    content_parts.append(f"Item: {meta['name_en']}")
                if meta["name_ar"]:
                    content_parts.append(f"الوجبة: {meta['name_ar']}")
                if meta["description_en"]:
                    content_parts.append(f"Description: {meta['description_en']}")
                if meta["description_ar"]:
                    content_parts.append(f"الوصف: {meta['description_ar']}")
                if meta["price"]:
                    content_parts.append(f"Price: {meta['price']}")
                normalized_alg_hits.append({
                    "content": " | ".join(content_parts) if content_parts else meta["name_ar"] or meta["name_en"],
                    "metadata": meta
                })

            merged_hits = sem_hits + normalized_alg_hits
            for h in merged_hits:
                meta = h.get("metadata", {})
                key = (meta.get("type"), meta.get("item_id") or meta.get("restaurant_id"), h.get("content"))
                if key in seen:
                    continue
                seen.add(key)
                if budget_egp is not None:
                    try:
                        price_val = float(meta.get("price", 0) or 0)
                    except Exception:
                        price_val = 0
                    if price_val and price_val > budget_egp:
                        continue
                results.append(h)
                if len(results) >= k:
                    break
            if len(results) >= k:
                break
        print(f"✅ recommend_time_based_suggestions collected {len(results)} results")
        return results
    except Exception as e:
        print(f"⚠ recommend_time_based_suggestions error: {e}")
        return []
restaurant_data = {}
try:
    for doc in db.collection("categories").where('where', '==', 'quweisna').stream():
        restaurant_data[doc.id] = doc.to_dict()
except Exception as e:
    print(f"⚠ Error loading restaurant data: {e}")

restaurant_name_to_id = {}
for rid, rdoc in restaurant_data.items():
    names = set()
    if rdoc.get("name_en"):
        names.add(normalize_arabic(rdoc["name_en"].strip().lower()))
    if rdoc.get("name_ar"):
        names.add(normalize_arabic(rdoc["name_ar"].strip().lower()))
    for n in names:
        restaurant_name_to_id[n] = rid 
