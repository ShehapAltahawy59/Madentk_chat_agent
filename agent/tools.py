from typing import List, Optional, Literal
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os
import json
import numpy as np
from sentence_transformers import CrossEncoder
from fuzzywuzzy import process
import unicodedata
import random
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

# Load environment variables
load_dotenv()
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
    # Accept either a file path or an inline JSON string
    if os.path.isfile(cred_env):
        cred_obj = credentials.Certificate(cred_env)
    else:
        try:
            cred_info = json.loads(cred_env)
            cred_obj = credentials.Certificate(cred_info)
        except json.JSONDecodeError:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS must be a valid file path or a JSON string."
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

# --- Vectorstore Setup with Summarized Chunks ---
device = 'cpu'
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': device}
)
chroma_path = os.environ.get("CHROMA_DB_DIR", "chroma_db")
collection_name = "food_data"

docs: List[Document] = []
vectorstore = None

if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=chroma_path
        )
    except Exception as e:
        print(f"⚠ Error loading vectorstore: {e}")
        vectorstore = None
else:
    restaurant_data = {}
    try:
        for doc in db.collection("categories").where('where', '==', 'quweisna').stream():
            restaurant_data[doc.id] = doc.to_dict()
    except Exception as e:
        print(f"⚠ Error fetching restaurant data: {e}")

    # Summarize items
    for doc in db.collection("items").stream():
        item = doc.to_dict()
        rest_id = item.get("item_cat")
        if rest_id not in restaurant_data:
            continue
        restaurant = restaurant_data[rest_id]

        item_summary = f"Item: {normalize_arabic(item.get('name_en', ''))} / {normalize_arabic(item.get('name_ar', ''))} | " \
                       f"Price: {item.get('price', '')} | " \
                       f"Restaurant: {normalize_arabic(restaurant.get('name_en', ''))} / {normalize_arabic(restaurant.get('name_ar', ''))}"

        docs.append(Document(page_content=item_summary, metadata={
            "type": "item",
            "item_id": item.get("item_id", ""),
            "restaurant_id": rest_id,
            "restaurant_name": normalize_arabic(restaurant.get("name_ar", ""))
        }))

    # Summarize restaurants
    for rest_id, restaurant in restaurant_data.items():
        restaurant_summary = f"Restaurant: {normalize_arabic(restaurant.get('name_en', ''))} / {normalize_arabic(restaurant.get('name_ar', ''))} | " \
                             f"Description: {normalize_arabic(restaurant.get('desc_en', '') or restaurant.get('desc_ar', ''))} | " \
                             f"City: {restaurant.get('where', '')}"

        docs.append(Document(page_content=restaurant_summary, metadata={
            "type": "restaurant",
            "restaurant_id": rest_id,
            "restaurant_name": normalize_arabic(restaurant.get("name_ar", ""))
        }))

    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=chroma_path
        )
        vectorstore.persist()
    except Exception as e:
        print(f"⚠ Error creating vectorstore: {e}")
        vectorstore = None

# Semantic Search Tool
def search_semantic(query: str, scope: Optional[Literal["item", "restaurant"]] = None, k: int = 20) -> List[dict]:
    if not vectorstore:
        print("⚠ Vectorstore not initialized, returning empty results")
        return []
    try:
        normalized_query = normalize_arabic(query)
        filters = {}
        if scope:
            filters["type"] = {"$eq": scope}
        docs_local = vectorstore.max_marginal_relevance_search(normalized_query, k=k, fetch_k=40, filter=filters)
        if docs_local:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
            pairs = [(normalized_query, doc.page_content) for doc in docs_local]
            scores = reranker.predict(pairs)
            reranked = [docs_local[i] for i in np.argsort(scores)[::-1]]
            docs_local = reranked[:k]
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
