# Docker Setup and Usage

This document explains how to build and run the API Pre Data Analysis service using Docker.

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose V2

## Quick Start

### Development Mode

For development with hot reloading:

```bash
# Build and run in development mode
docker-compose -f docker-compose.dev.yml up --build

# Run in background
docker-compose -f docker-compose.dev.yml up -d --build
```

### Production Mode

For production deployment:

```bash
# Build and run the API only
docker-compose up --build

# Run with nginx reverse proxy
docker-compose --profile production up --build

# Run in background
docker-compose up -d --build
```

## Available Services

### API Service
- **Port**: 8000
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs

### Nginx Reverse Proxy (Production Profile)
- **Port**: 80 (HTTP)
- **Port**: 443 (HTTPS - requires SSL configuration)

## Docker Commands

### Building the Image

```bash
# Build the Docker image
docker build -t api-pre-data-analysis .

# Build with custom tag
docker build -t api-pre-data-analysis:v1.0.0 .
```

### Running the Container

```bash
# Run the container directly
docker run -p 8000:8000 api-pre-data-analysis

# Run with environment variables
docker run -p 8000:8000 -e ENVIRONMENT=production api-pre-data-analysis

# Run in background
docker run -d -p 8000:8000 --name api-container api-pre-data-analysis
```

### Managing Containers

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs api
docker-compose logs -f api  # Follow logs

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart services
docker-compose restart
```

## Configuration

### Environment Variables

The following environment variables can be configured:

- `PYTHONPATH`: Python path (default: /app)
- `ENVIRONMENT`: Environment mode (development/production)

### Volume Mounts

- `api_temp`: Temporary file storage for the API
- Source code mount (development only): `.:/app`

### Health Checks

The container includes health checks that verify the API is responding:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Start Period**: 40 seconds

## Production Deployment

### With Nginx Reverse Proxy

1. Configure your domain in `nginx.conf`
2. Add SSL certificates if using HTTPS
3. Run with production profile:

```bash
docker-compose --profile production up -d --build
```

### SSL Configuration (Optional)

1. Place your SSL certificates in an `ssl/` directory:
   - `ssl/cert.pem`
   - `ssl/key.pem`

2. Uncomment the HTTPS server block in `nginx.conf`

3. Update the volume mount in `docker-compose.yml`

### File Upload Limits

The nginx configuration is set to handle file uploads up to 100MB. Adjust `client_max_body_size` in `nginx.conf` if needed.

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Check what's using port 8000
   netstat -tulpn | grep :8000
   # Stop the service or change port in docker-compose.yml
   ```

2. **Permission denied**:
   ```bash
   # Ensure Docker daemon is running
   sudo systemctl start docker
   ```

3. **Container fails health check**:
   ```bash
   # Check container logs
   docker-compose logs api
   
   # Manually test health endpoint
   docker exec -it api_pre_data_analysis python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
   ```

### Debugging

```bash
# Enter running container
docker exec -it api_pre_data_analysis bash

# Check container resource usage
docker stats api_pre_data_analysis

# Inspect container details
docker inspect api_pre_data_analysis
```

## Development Tips

1. Use `docker-compose.dev.yml` for development with hot reloading
2. Code changes are automatically reflected without rebuilding
3. Use `docker-compose logs -f api-dev` to monitor development logs
4. Install additional packages by updating `pyproject.toml` and rebuilding

## Security Considerations

- The container runs as a non-root user (uid: 1000)
- Only necessary ports are exposed
- No sensitive data is included in the image
- Consider using secrets management for production deployments
