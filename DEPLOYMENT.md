# Deployment Guide
## FRA Diagnostics Platform - SIH 2025 PS 25190

This guide covers various deployment scenarios for the Transformer FRA Diagnostics Platform.

---

## 📋 Prerequisites

- Python 3.9+ (3.10 recommended)
- Docker 20.10+ (for containerized deployment)
- Git
- 4GB+ RAM
- 10GB+ disk space

---

## 🚀 Local Development

### 1. Clone Repository
```bash
git clone <repository-url>
cd "SIH PS2"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools
```

### 4. Set Environment Variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Run Application
```bash
streamlit run app.py
```

Access at: http://localhost:8501

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Build
```bash
# Build image
docker build -t fra-diagnostics:latest .

# Run container
docker run -d \
  --name fra-diagnostics \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  -e FRA_LOG_LEVEL=INFO \
  fra-diagnostics:latest

# View logs
docker logs -f fra-diagnostics

# Stop container
docker stop fra-diagnostics
docker rm fra-diagnostics
```

---

## ☁️ Production Deployment

### Environment Configuration

Create a `.env` file with production settings:

```env
# Logging
FRA_LOG_LEVEL=INFO
FRA_LOG_DIR=/app/logs

# Directories
FRA_MODEL_DIR=/app/models
FRA_TEMP_DIR=/app/temp
FRA_DATA_DIR=/app/data

# Security
FRA_REQUIRE_AUTH=true
FRA_ENABLE_RATE_LIMIT=true

# Performance
FRA_NUM_WORKERS=4
FRA_DEVICE=cpu
FRA_ENABLE_PROFILING=false
```

### Using Docker Compose (Recommended)

1. **Update docker-compose.yml** with production values
2. **Deploy:**
```bash
docker-compose -f docker-compose.yml up -d
```

### Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fra-diagnostics
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fra-diagnostics
  template:
    metadata:
      labels:
        app: fra-diagnostics
    spec:
      containers:
      - name: fra-diagnostics
        image: fra-diagnostics:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: fra-models-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: fra-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: fra-diagnostics
spec:
  selector:
    app: fra-diagnostics
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s/
```

---

## 🔒 Security Best Practices

### 1. Use HTTPS in Production
Set up reverse proxy (nginx/traefik) with SSL certificates.

### 2. Enable Authentication
```env
FRA_REQUIRE_AUTH=true
```

Configure Streamlit authentication in `app.py`.

### 3. Rate Limiting
```env
FRA_ENABLE_RATE_LIMIT=true
```

### 4. File Upload Restrictions
- Max file size: 50MB (configurable)
- Allowed extensions: .csv, .xml, .txt, .dat
- Magic byte validation enabled

### 5. Run as Non-Root User
Dockerfile already configured with `frauser` (UID 1000).

---

## 📊 Monitoring

### Health Check Endpoint
```bash
curl http://localhost:8501/_stcore/health
```

### Logs
```bash
# Docker logs
docker logs -f fra-diagnostics

# Local logs
tail -f logs/fra_app_*.log
```

### Metrics (Future Enhancement)
Add Prometheus exporter for metrics:
- Request count
- Processing time
- Error rates
- Model inference latency

---

## 🔧 Troubleshooting

### Issue: Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501
kill -9 <PID>
```

### Issue: Model Files Not Found
```bash
# Ensure models directory is mounted correctly
docker exec fra-diagnostics ls -la /app/models
```

### Issue: Out of Memory
- Increase Docker memory limit in docker-compose.yml
- Reduce batch size in config.py
- Use CPU instead of GPU if OOM on GPU

### Issue: Permission Denied
```bash
# Fix file permissions
sudo chown -R 1000:1000 models/ logs/ temp/ data/
```

---

## 🔄 Updates and Rollback

### Update to New Version
```bash
# Pull latest code
git pull origin main

# Rebuild Docker image
docker-compose build

# Restart with new image
docker-compose up -d
```

### Rollback
```bash
# List available images
docker images fra-diagnostics

# Run previous version
docker run -d \
  --name fra-diagnostics \
  -p 8501:8501 \
  fra-diagnostics:<previous-tag>
```

---

## 📞 Support

For issues or questions:
- Check logs first: `docker logs fra-diagnostics`
- Review health check: `curl localhost:8501/_stcore/health`
- Consult README.md for configuration options

---

## 🎯 Performance Optimization

### 1. Enable Model Caching
Models are cached by default using `@st.cache_resource`.

### 2. Use Production WSGI Server
For high-traffic deployments, consider using gunicorn with streamlit.

### 3. Database for Results
For result persistence, add PostgreSQL:
```yaml
# In docker-compose.yml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fra_diagnostics
      POSTGRES_USER: frauser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
```

---

## ✅ Deployment Checklist

Before production deployment:

- [ ] Environment variables configured
- [ ] SSL/TLS certificates installed
- [ ] Authentication enabled
- [ ] Rate limiting enabled
- [ ] Model files present in models/
- [ ] Backups configured
- [ ] Monitoring set up
- [ ] Health checks working
- [ ] Resource limits set
- [ ] Security scan passed
- [ ] Load testing completed
- [ ] Documentation reviewed
