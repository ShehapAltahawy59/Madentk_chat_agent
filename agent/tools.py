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

# --- Arabic Normalization ---
def normalize_arabic(text: str) -> str:
    if not text:
        return text
    text = text.strip().lower()
    replacements = {
        'ي': 'ى',
        'أ': 'ا',
        'إ': 'ا',
        'آ': 'ا',
        'ة': 'ه',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize('NFKC', text)
    return text

# --- Firestore Tools ---
REQUIRED_FIELDS = {
    "addressid": "",
    "cancelreason": "",
    "delivery": {"name": "", "phone": ""},
    "deliveryCost": 0,
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
                "deliveryCost": 15, # constant
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
        doc = db.collection("categories").document(restaurant_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        print(f"⚠ Error fetching restaurant: {e}")
        return None

def get_item_by_id(item_id: str, restaurant_id: Optional[str] = None) -> dict:
    try:
        query = db.collection("items").where("item_id", "==", item_id)
        if restaurant_id:
            query = query.where("item_cat", "==", restaurant_id)
        doc = next(query.stream(), None)
        if doc:
            return doc.to_dict()
        return None
    except Exception as e:
        print(f"⚠ Error fetching item: {e}")
        return None

def get_items_in_restaurant(restaurant_id: str) -> List[dict]:
    try:
        query = db.collection("items").where("item_cat", "==", restaurant_id).stream()
        return [doc.to_dict() for doc in query]
    except Exception as e:
        print(f"⚠ Error fetching items: {e}")
        return []

def search_restaurant_by_name(name: str) -> List[dict]:
    try:
        restaurants = []
        for doc in db.collection("categories").where("where", "==", "quweisna").stream():
            restaurants.append(doc.to_dict())
        normalized_name = normalize_arabic(name)
        matches = []
        for rest in restaurants:
            name_en = normalize_arabic(rest.get("name_en", "").strip().lower())
            name_ar = normalize_arabic(rest.get("name_ar", "").strip().lower())
            score_en = process.extractOne(normalized_name, [name_en])[1] if name_en else 0
            score_ar = process.extractOne(normalized_name, [name_ar])[1] if name_ar else 0
            if score_en >= 80 or score_ar >= 80:
                matches.append(rest)
        print(f"Direct search for '{name}' (normalized: '{normalized_name}') found: {[r.get('name_ar', '') for r in matches]}")
        return matches
    except Exception as e:
        print(f"⚠ Error searching restaurants by name: {e}")
        return []

# --- Vectorstore Setup with Startup Loading ---
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
    
    if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
        try:
            print("🔄 Loading vectorstore...")
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=chroma_path
            )
            print("✅ Vectorstore loaded from existing directory")
        except Exception as e:
            print(f"⚠ Error loading vectorstore: {e}")
            vectorstore = None
    else:
        print(f"⚠ Chroma DB directory not found at {chroma_path}, vectorstore not initialized")
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

# Semantic Search Tool
def search_semantic(query: str, scope: Optional[Literal["item", "restaurant"]] = None, k: int = 20) -> List[dict]:
    vs = get_vectorstore()
    if not vs:
        print("⚠ Vectorstore not available for semantic search")
        return []
    
    try:
        normalized_query = normalize_arabic(query)
        filters = {}
        if scope:
            filters["type"] = {"$eq": scope}
        docs_local = vs.max_marginal_relevance_search(normalized_query, k=k, fetch_k=40, filter=filters)
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
