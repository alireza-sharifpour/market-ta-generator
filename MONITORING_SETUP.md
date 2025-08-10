# Grafana Monitoring Stack Setup Guide

This guide provides step-by-step instructions for setting up a comprehensive monitoring stack with Grafana, Prometheus, and Loki for the Market TA Generator application.

## Overview

The monitoring stack includes:
- **Grafana**: Visualization and dashboards
- **Prometheus**: Metrics collection and storage
- **Loki**: Log aggregation and storage
- **Promtail**: Log shipping agent
- **Node Exporter**: System metrics collection
- **cAdvisor**: Container metrics collection

## Prerequisites

- Docker and Docker Compose installed on your server
- SSH access to your server: `ssh -i ~/.ssh/github_action_key alireza@38.54.76.46`
- Sufficient disk space for metrics and logs storage

## Deployment Steps

### 1. Upload Files to Server

```bash
# Copy the entire project to your server
rsync -avz -e "ssh -i ~/.ssh/github_action_key" /path/to/market-ta-generator/ alireza@38.54.76.46:~/market-ta-generator/

# Or copy just the monitoring configuration
scp -i ~/.ssh/github_action_key -r monitoring/ alireza@38.54.76.46:~/market-ta-generator/
scp -i ~/.ssh/github_action_key docker-compose.yml alireza@38.54.76.46:~/market-ta-generator/
```

### 2. Connect to Server and Start Services

```bash
# SSH into your server
ssh -i ~/.ssh/github_action_key alireza@38.54.76.46

# Navigate to project directory
cd market-ta-generator

# Start the monitoring stack
docker-compose up -d

# Check service status
docker-compose ps
```

### 3. Verify Services

Check that all services are running:

```bash
# Check container status
docker-compose ps

# Check logs for any issues
docker-compose logs grafana
docker-compose logs prometheus
docker-compose logs loki
```

## Access Points

Once deployed, you can access the following services:

- **Grafana Dashboard**: http://38.54.76.46:3000
  - Username: `admin`
  - Password: `MarketTA2025!Secure#Grafana`
  
- **Prometheus**: http://38.54.76.46:9090
- **Loki**: http://38.54.76.46:3100
- **Your Application**: http://38.54.76.46:8000
- **Node Exporter**: http://38.54.76.46:9100
- **cAdvisor**: http://38.54.76.46:8080

## Configuration Details

### Prometheus Configuration

The Prometheus configuration (`monitoring/prometheus/prometheus.yml`) includes scrape targets for:
- The FastAPI application metrics at `/metrics`
- Node Exporter for system metrics
- cAdvisor for container metrics

### Loki Configuration

Loki is configured to:
- Store logs in the `/loki` directory inside the container
- Accept logs from Promtail
- Provide a query interface for Grafana

### Grafana Setup

Grafana is pre-configured with:
- Prometheus as the default data source
- Loki as a secondary data source
- A FastAPI dashboard showing key application metrics

## Monitoring Features

### Application Metrics

The FastAPI application automatically exposes these metrics:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `http_requests_inprogress`: Active requests
- `http_request_size_bytes`: Request payload sizes
- `http_response_size_bytes`: Response payload sizes

### System Metrics

Node Exporter provides:
- CPU usage and load
- Memory utilization
- Disk I/O and space
- Network traffic

### Container Metrics

cAdvisor provides:
- Container resource usage
- Container performance metrics
- Docker container statistics

### Log Aggregation

Promtail collects logs from:
- Docker container logs
- System logs
- Application-specific logs

## Dashboard Usage

### Pre-built Dashboard

The FastAPI dashboard includes:
- HTTP request rate over time
- Response time percentiles (50th, 95th, 99th)
- Current requests in progress
- Requests per minute counter

### Custom Queries

Use these PromQL queries in Grafana:

```promql
# Request rate
rate(http_requests_total{job="market-ta-generator"}[5m])

# Response time 95th percentile
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# CPU usage
rate(container_cpu_usage_seconds_total{name="market-ta-generator"}[5m])
```

## Troubleshooting

### Common Issues

1. **Services won't start**
   ```bash
   # Check logs
   docker-compose logs
   
   # Check disk space
   df -h
   ```

2. **No metrics in Grafana**
   ```bash
   # Verify Prometheus can reach the app
   curl http://localhost:8000/metrics
   
   # Check Prometheus targets
   # Visit http://38.54.76.46:9090/targets
   ```

3. **No logs in Loki**
   ```bash
   # Check Promtail status
   docker-compose logs promtail
   
   # Verify Loki is accessible
   curl http://localhost:3100/ready
   ```

### Log Locations

- Container logs: `/var/lib/docker/containers/`
- Prometheus data: Docker volume `prometheus_data`
- Loki data: Docker volume `loki_data`
- Grafana data: Docker volume `grafana_data`

## Log Retention and Storage Management

### Automatic Log Cleanup Configuration

The monitoring stack is configured with comprehensive log retention policies to prevent disk space issues:

#### **Systemd Journal Limits**
- **Maximum size**: 1GB (down from ~3.3GB default)
- **Rotation**: Weekly (instead of monthly)
- **Retention**: 2 months maximum
- **Compression**: Enabled to save space
- **Free space**: Maintains 500MB minimum

Configuration applied via `scripts/configure_systemd_journal.sh`:
```bash
# Run this script to configure systemd journal limits
sudo ./scripts/configure_systemd_journal.sh
```

#### **Loki Log Retention**
- **Retention period**: 7 days (168h)
- **Compactor**: Enabled with automatic deletion
- **Delete delay**: 2 hours (safety buffer)
- **Working directory**: `/loki/compactor`
- **Storage**: Filesystem-based with delete request tracking

Configuration in `monitoring/loki/loki-config.yml`:
```yaml
limits_config:
  retention_period: 168h  # 7 days
  max_query_lookback: 168h

compactor:
  working_directory: /loki/compactor
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
  delete_request_store: filesystem
```

#### **Docker Container Log Rotation**
All containers are configured with bounded log growth:

- **Application container**: 10MB × 5 files = 50MB maximum
- **Monitoring containers** (Prometheus, Grafana, Loki): 20MB × 6 files = 120MB maximum each

Configuration in `docker-compose.yml`:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "20m"  # 10m for app container
    max-file: "6"    # 5 for app container
```

#### **Prometheus Data Retention**
- **Time-based retention**: 15 days
- **Storage optimization**: TSDB format with compression
- **Volume management**: Automatic cleanup of old data

### Storage Usage Monitoring

#### Expected Storage Consumption:
- **Systemd journals**: ~1GB (automatically managed)
- **Loki logs**: ~200MB (7-day rolling window)
- **Prometheus metrics**: ~400MB (15-day retention)
- **Grafana data**: ~40MB (dashboards and config)
- **Container logs**: 50-120MB per container (bounded)

#### **Total Expected Usage**: ~2-3GB (vs previous 15+ GB unbounded growth)

### Maintenance Commands

#### **Check Current Storage Usage**
```bash
# Overall disk usage
df -h

# Systemd journal usage
sudo journalctl --disk-usage

# Docker system usage
docker system df

# Individual volume sizes
docker exec loki du -sh /loki
docker exec prometheus du -sh /prometheus
docker exec grafana du -sh /var/lib/grafana
```

#### **Manual Cleanup (if needed)**
```bash
# Clean systemd journals (emergency cleanup)
sudo journalctl --vacuum-size=500M
sudo journalctl --vacuum-time=7d

# Clean Docker build cache
docker builder prune -a -f

# Clean unused images
docker image prune -a -f
```

### Backup

```bash
# Backup Grafana dashboards and settings
docker run --rm -v market-ta-generator_grafana_data:/data -v $(pwd):/backup alpine tar czf /backup/grafana-backup.tar.gz -C /data .

# Backup Prometheus data
docker run --rm -v market-ta-generator_prometheus_data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz -C /data .
```

### Updates

```bash
# Update images
docker compose pull

# Restart services with new log limits
docker compose down
docker compose up -d
```

### Automated Cleanup

The following cleanup happens automatically:
1. **Systemd journals**: Rotate weekly, delete after 2 months, maintain 1GB max
2. **Loki logs**: Delete logs older than 7 days every 2 hours
3. **Docker container logs**: Rotate when files exceed size limits
4. **Prometheus data**: Delete metrics older than 15 days

## Security Considerations

1. **Change default passwords**
   - Grafana admin password is set to `MarketTA2025!Secure#Grafana`
   - Consider using environment variables for sensitive data

2. **Network security**
   - Use a reverse proxy with SSL/TLS
   - Restrict access to monitoring ports using firewall rules

3. **Data retention**
   - Configure appropriate retention policies for logs and metrics
   - Monitor disk usage regularly

## Performance Tuning

### For High Traffic

1. **Increase Prometheus retention**
   ```yaml
   command:
     - '--storage.tsdb.retention.time=30d'  # Increase from 15d
   ```

2. **Optimize Loki configuration**
   ```yaml
   # Add to loki-config.yml
   limits_config:
     max_query_parallelism: 32
     max_streams_per_user: 10000
   ```

3. **Adjust scrape intervals**
   ```yaml
   # In prometheus.yml
   scrape_interval: 30s  # Increase from 15s for less load
   ```

## Next Steps

1. Set up alerting rules in Prometheus
2. Configure notification channels in Grafana
3. Create custom dashboards for business metrics
4. Implement log-based alerting in Loki
5. Set up automated backups

For additional help, refer to the official documentation:
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Loki Documentation](https://grafana.com/docs/loki/)