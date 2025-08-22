# 🍕 SmartFood Agent - Arabic Food Ordering Assistant

A smart food ordering assistant powered by AI that understands Egyptian Arabic and helps users order food through natural conversation.

## 🌟 Features

- **🤖 AI-Powered Conversations**: Natural Arabic language processing using AutoGen
- **🔍 Smart Search**: Semantic and fuzzy search for restaurants and food items
- **🌍 Bilingual Support**: Works with both Arabic and English
- **📱 Modern UI**: Clean Streamlit interface
- **☁️ Cloud Deployed**: Ready-to-use on Google Cloud Run
- **🔒 Secure**: Firebase authentication and data protection

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Google Cloud account (for deployment)
- Firebase project

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Madentk_chat_agent
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.backend.txt
   pip install -r requirements.streamlit.txt
   ```

4. **Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

5. **Access the application**
   - Backend API: http://localhost:8080
   - Streamlit UI: http://localhost:8501

## 🏗️ Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with Uvicorn
- **AI Engine**: AutoGen with Gemini 2.5 Flash
- **Database**: Firebase Firestore
- **Search**: Chroma DB with semantic search
- **Models**: Sentence Transformers + CrossEncoder

### Frontend (Streamlit)
- **Framework**: Streamlit
- **Features**: Real-time chat interface
- **Responsive**: Works on mobile and desktop

### Infrastructure
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **Deployment**: Google Cloud Run
- **CI/CD**: GitHub Actions

## 🛠️ Tools & Technologies

### AI & ML
- **AutoGen**: Conversational AI framework
- **Sentence Transformers**: Multilingual embeddings
- **CrossEncoder**: Reranking for better search results
- **Fuzzy Matching**: Handles typos and variations

### Backend
- **FastAPI**: Modern Python web framework
- **Firebase Admin**: Database and authentication
- **Chroma DB**: Vector database for semantic search
- **LangChain**: LLM integration

### Frontend
- **Streamlit**: Rapid web app development
- **Requests**: HTTP client for API calls

### DevOps
- **Docker**: Containerization
- **Google Cloud Run**: Serverless deployment
- **GitHub Actions**: Automated CI/CD

## 📁 Project Structure

```
Madentk_chat_agent/
├── agent/                 # AI agent components
│   ├── agent.py          # Main agent configuration
│   ├── client.py         # Model client setup
│   ├── prompt.py         # System prompts
│   └── tools.py          # Agent tools and functions
├── routes/               # API routes
│   └── chat.py          # Chat endpoint
├── app.py               # FastAPI application
├── streamlit_app.py     # Streamlit UI
├── requirements*.txt    # Dependencies
├── Dockerfile.*         # Container configurations
├── docker-compose.yml   # Local development
└── .github/             # CI/CD workflows
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
Gemini_API_KEY=your_gemini_api_key
HUGGINGFACE_HUB_TOKEN=your_hf_token

# Firebase
GOOGLE_APPLICATION_CREDENTIALS=your_firebase_credentials

# Database
CHROMA_DB_DIR=chroma_db

# URLs
CHAT_API_BASE_URL=http://localhost:8080
```

### Firebase Setup

1. Create a Firebase project
2. Enable Firestore database
3. Create service account and download credentials
4. Add collections: `users`, `categories`, `items`, `orders`

## 🚀 Deployment

### Google Cloud Run

1. **Set up Google Cloud**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Deploy using GitHub Actions**
   - Push to main branch
   - GitHub Actions will build and deploy automatically

3. **Manual deployment**
   ```bash
   # Build images
   docker-compose build
   
   # Deploy to Cloud Run
   gcloud run deploy smartfood-backend --image gcr.io/PROJECT_ID/backend
   gcloud run deploy smartfood-frontend --image gcr.io/PROJECT_ID/frontend
   ```

## 🧪 Testing

### Local Testing
```bash
# Test backend
python -c "import app; print('✅ Backend ready')"

# Test agent tools
python -c "from agent.tools import get_item_by_name; print('✅ Tools ready')"

# Test API
curl http://localhost:8080/health
```

### API Endpoints

- `GET /health` - Health check
- `GET /` - API info
- `POST /chat` - Chat with AI agent
- `GET /debug` - Debug information

## 🤖 AI Agent Tools

The agent has access to these tools:

1. **`get_user_by_id`** - Fetch user data
2. **`insert_order`** - Place food orders
3. **`search_semantic`** - Semantic search for food/restaurants
4. **`get_restaurant_by_id`** - Get restaurant details
5. **`get_item_by_id`** - Get food item details
6. **`get_item_by_name`** - Fuzzy search for items by name
7. **`get_items_in_restaurant`** - List restaurant menu
8. **`search_restaurant_by_name`** - Find restaurants by name
9. **`get_active_user_id`** - Get current user context

## 🌍 Language Support

### Arabic Features
- **Egyptian Arabic**: Natural conversation support
- **Text Normalization**: Handles Arabic character variations
- **Fuzzy Matching**: Tolerates typos and variations
- **Bilingual Search**: Works with Arabic and English

### Example Conversations
```
User: "عاوز برجر"
Agent: "ممتاز! لقيت كذا نوع برجر. في برجر لحم، برجر دجاج، برجر نباتي. إيه النوع اللي عايزه؟"

User: "I want pizza"
Agent: "Great! I found several pizza options. We have Margherita, Pepperoni, and Vegetarian. Which one would you like?"
```

## 🔒 Security

- **Firebase Authentication**: Secure user management
- **Environment Variables**: Sensitive data protection
- **Input Validation**: Sanitized user inputs
- **Rate Limiting**: API protection
- **HTTPS**: Secure communication

## 📊 Performance

- **Model Loading**: Optimized startup with lazy loading
- **Memory Management**: Efficient resource usage
- **Response Time**: Fast AI responses
- **Scalability**: Cloud-native architecture

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Increase Docker memory
   docker-compose up --build --memory=4g
   ```

2. **Firebase Connection**
   ```bash
   # Check credentials
   python -c "from agent.tools import db; print('✅ Firebase connected')"
   ```

3. **Model Loading**
   ```bash
   # Check model status
   curl http://localhost:8080/debug
   ```

### Logs
```bash
# Backend logs
docker-compose logs backend

# Frontend logs
docker-compose logs streamlit
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AutoGen Team**: For the amazing conversational AI framework
- **Firebase**: For the robust backend services
- **Streamlit**: For the beautiful UI framework
- **Google Cloud**: For the reliable cloud infrastructure

---

**Made with ❤️ for the Arabic-speaking community**
