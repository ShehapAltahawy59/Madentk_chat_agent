# Madentk Chat Agent

## Algolia Search

Set these environment variables to enable Algolia full-text and fuzzy-typo tolerant search:

- `ALGOLIA_APP_ID`
- `ALGOLIA_API_KEY`
- `ALGOLIA_CATEGORIES_INDEX` (default: `categories_index`)
- `ALGOLIA_ITEMS_INDEX` (default: `items_index`)

The backend will prefer Algolia for `search_restaurant_by_name` and `get_item_by_name`. Ensure your indices contain fields used for filtering: `where`, and for items either `item_cat` or `restaurant_id`.

SmartFoodAgent is a location-aware food ordering/chat assistant built with FastAPI (backend) and Streamlit (UI). It integrates Firebase (Firestore) for operational data and Chroma DB for semantic search with multilingual embeddings and reranking.

## ✨ Features
- Location-scoped search using `where` (e.g., `quweisna`, `AboHammad`, `KafrShokr`)
- Semantic search backed by Chroma DB (multilingual embeddings + optional reranking)
- Firestore data model: categories (restaurants), items, orders, users
- Robust fuzzy search for Arabic/English names
- Automatic vector DB creation on server start if not present
- Streamlit UI with location selector; always sends `where` to backend

## 🧱 Architecture
- Backend: FastAPI (`app.py`) with routes in `routes/`
- UI: Streamlit (`streamlit_app.py`)
- Agent tools and data access in `agent/tools.py`
- Vector DB store: Chroma DB persisted to directory `chroma_db/` (or `/data/chroma_db` in Docker)

## 📦 Requirements
- Python 3.10+
- Firebase Service Account credentials (JSON)
- Optional: Hugging Face token (for model downloads/rate limits)

## 🔧 Configuration
Create a `.env` file (for local dev) with:

```bash
# API Keys
Gemini_API_KEY=your_gemini_api_key
HUGGINGFACE_HUB_TOKEN=your_hf_token  # optional

# Firebase (required)
GOOGLE_APPLICATION_CREDENTIALS=path_or_json  # file path OR base64 OR raw JSON

# Database
CHROMA_DB_DIR=chroma_db

# URLs
CHAT_API_BASE_URL=http://localhost:8080
```

Notes about `GOOGLE_APPLICATION_CREDENTIALS`:
- You can provide a file path to a JSON file
- Or a base64-encoded JSON string
- Or a raw JSON string
The backend will auto-detect the format.

## 🗂️ Firestore Collections
Expected collections and key fields:
- `categories` (restaurants)
  - `id` (document ID), `name_en`, `name_ar`, `description_en`, `description_ar`, `where`
- `items`
  - `item_id`, `item_cat` (restaurant id), `name_en`, `name_ar`, `description_en`, `description_ar`, `price`
  - Note: items do NOT have a `where`; they inherit location from their restaurant
- `orders`
- `users`

## 🔎 Vector DB (Chroma)
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional if available)
- Collection name: `food_data`
- Persisted under `CHROMA_DB_DIR` (default `chroma_db/`)

### Auto create on server start
On backend startup, if the persist directory does not exist or the collection cannot be loaded, the code will create the vector DB from Firestore data and persist it.

- Restaurant docs include metadata: `type=restaurant`, `where`, ids/names/descriptions
- Item docs include metadata: `type=item`, `restaurant_id`, `where` (inherited from the item’s restaurant)

### Manual creation (optional)
You can recreate vector DB anytime via the helper script:

```bash
python create_vector_db.py
```

## ▶️ Run Locally
Install deps and run the backend + UI.

```bash
# Backend
pip install -r requirements.backend.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# UI
pip install -r requirements.streamlit.txt
streamlit run streamlit_app.py --server.port 8501
```

## 🧪 Quick Test
A quick script to test the deployed API is included:
```bash
python quick_test.py
```

## 🖥️ Streamlit UI
- Select your location in the sidebar (dropdown)
- The UI always sends the `where` field to the backend
- The chat history resets when changing location to avoid mixing contexts

## 🌐 API
### POST `/chat`
Request body:
```json
{
  "user_query": "مرحبا",
  "history": [["user", "assistant"], ...],
  "user_id": "optional",
  "where": "quweisna"
}
```
Response:
```json
{ "response": "..." }
```

Other routes (for local/dev):
- GET `/health`: health check
- GET `/debug`: debug info (if enabled)

## 🧠 Location-aware Search
- The backend stores active context (`user_id`, `where`) per request
- All data lookups respect `where`:
  - Restaurants are filtered by `where`
  - Items inherit `where` from their `item_cat` restaurant
- Semantic search in Chroma uses location-first retrieval, then optional in-memory scope filtering:
  - First retrieval: filter by `where`
  - Second filtering: if `scope` provided (`item` or `restaurant`), restrict results to that type

## 🐳 Docker
Build images and run with Docker Compose.

```bash
# Build & run
docker compose up --build

# Services
# - Backend on localhost:8000 (container 8080)
# - Streamlit on localhost:8501 (container 8080)
```

Docker specifics:
- Backend image sets `CHROMA_DB_DIR=/data/chroma_db` and mounts volume `chroma_data`
- Provide required env vars via `.env` or compose env

## ⚠️ Troubleshooting
- Firebase credentials errors:
  - Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid path, base64 string, or raw JSON
- Chroma DB filter error:
  - Chroma supports a single filter; we retrieve by `where` first, then optionally filter by `type` in-memory
- Model downloads slow/failing:
  - Provide `HUGGINGFACE_HUB_TOKEN`; ensure internet access from the environment
- No search results:
  - Check the selected location in the UI and that Firestore data has matching `where`
- Docker volume issues:
  - Remove/recreate the `chroma_data` volume if the persisted store becomes inconsistent

## 🧰 Key Files
- `agent/tools.py`: Firebase access, Vector DB creation/search, fuzzy/semantic tools
- `routes/chat.py`: Chat endpoint; sets active context (`set_active_user_id`, `set_active_where`)
- `app.py`: FastAPI app wiring and health/debug routes
- `streamlit_app.py`: UI with location selector; posts to `/chat`

## 📜 License
MIT
