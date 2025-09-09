# Chat Agent Deployment Guide

This project is configured for automatic deployment to your server using GitHub Actions.

## Deployment Setup

### Services Created

1. **Backend API Service** (`madentk-chat-agent.service`)
   - Port: 8001 (to avoid conflicts with existing services on 8000/8080)
   - Runs FastAPI application
   - Environment file: `/etc/madentk-chat-agent.env`

2. **Streamlit UI Service** (`madentk-chat-agent-ui.service`)
   - Port: 8502 (to avoid conflicts with existing services)
   - Runs Streamlit web interface
   - Environment file: `/etc/madentk-chat-agent-ui.env`

### Required GitHub Secrets

Make sure these secrets are configured in your GitHub repository:

- `SERVER_USER`: Username for server access
- `SERVER_PASS`: Password for server access
- `GOOGLE_APPLICATION_CREDENTIALS`: Google service account credentials
- `Gemini_API_KEY`: Google Gemini API key
- `HUGGINGFACE_HUB_TOKEN`: Hugging Face Hub token

### Deployment Workflows

1. **`.github/workflows/deploy-to-server.yml`**
   - Deploys the FastAPI backend service
   - Runs on push to main branch
   - Creates systemd service for automatic startup

2. **`.github/workflows/deploy-streamlit-to-server.yml`**
   - Deploys the Streamlit UI service
   - Runs on push to main branch
   - Creates separate systemd service for UI

### Server Configuration

The deployment will:

1. Clone/update the repository to `/Madentk_chat_agent`
2. Create Python virtual environment (`madentk-chat-agent-venv`)
3. Install requirements from `requirements.txt`
4. Create systemd services for both backend and UI
5. Set up environment files with secrets
6. Start services automatically

### Service Management

After deployment, you can manage the services using:

```bash
# Check service status
sudo systemctl status madentk-chat-agent.service
sudo systemctl status madentk-chat-agent-ui.service

# View logs
sudo journalctl -u madentk-chat-agent.service -f
sudo journalctl -u madentk-chat-agent-ui.service -f

# Restart services
sudo systemctl restart madentk-chat-agent.service
sudo systemctl restart madentk-chat-agent-ui.service

# Stop services
sudo systemctl stop madentk-chat-agent.service
sudo systemctl stop madentk-chat-agent-ui.service
```

### Access URLs

After successful deployment:

- **Backend API**: http://173.212.251.191:8001
- **Streamlit UI**: http://173.212.251.191:8502

### Port Configuration

- Backend API: 8001 (avoiding conflicts with existing services on 8000/8080)
- Streamlit UI: 8502 (avoiding conflicts with existing services)

### Environment Variables

The deployment automatically creates environment files:

- `/etc/madentk-chat-agent.env` - Backend service environment
- `/etc/madentk-chat-agent-ui.env` - UI service environment

### Manual Deployment

If you need to deploy manually:

1. SSH into your server
2. Navigate to the project directory
3. Run the start scripts:
   ```bash
   # For backend
   ./start.sh
   
   # For UI (in another terminal)
   ./start-ui.sh
   ```

### Troubleshooting

1. **Check service status**: `sudo systemctl status madentk-chat-agent.service`
2. **View logs**: `sudo journalctl -u madentk-chat-agent.service -f`
3. **Check port usage**: `sudo netstat -tlnp | grep :8001`
4. **Restart services**: `sudo systemctl restart madentk-chat-agent.service`
