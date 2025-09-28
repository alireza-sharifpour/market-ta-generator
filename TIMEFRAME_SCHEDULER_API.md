# Multi-Timeframe Volume Scheduler API

The enhanced volume analysis scheduler now supports **multiple concurrent timeframe-based cron jobs**. You can run different analysis schedules simultaneously (5min, 15min, 1hour, etc.) and manage them independently.

## 🚀 Quick Start

### Start the FastAPI Service
```bash
cd /media/mra/w/w/amiram/projects/market-ta-generator
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Example: Start Multiple Timeframes
```bash
# Start 5-minute analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/5m/start"

# Start 1-hour analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/1h/start"

# Start 4-hour analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/4h/start"
```

## 📋 Available Timeframes

| Timeframe | Interval | Display Name | Cron Schedule |
|-----------|----------|--------------|---------------|
| `5m` | 5 minutes | "5 Minutes" | `*/5 * * * *` |
| `1h` | 1 hour | "1 Hour" | `0 * * * *` |
| `4h` | 4 hours | "4 Hours" | `0 */4 * * *` |
| `1d` | 1 day | "1 Day" | `0 0 * * *` |

## 🔄 New API Endpoints

### 1. Get Available Timeframes

**Endpoint:** `GET /api/v1/scheduler/timeframes`

```bash
curl "http://localhost:8000/api/v1/scheduler/timeframes"
```

**Response:**
```json
{
  "status": "success",
  "available_timeframes": {
    "5m": {
      "interval_minutes": 5,
      "display_name": "5 Minutes"
    },
    "1h": {
      "interval_minutes": 60,
      "display_name": "1 Hour"
    },
    "4h": {
      "interval_minutes": 240,
      "display_name": "4 Hours"
    },
    "1d": {
      "interval_minutes": 1440,
      "display_name": "1 Day"
    }
  },
  "total_count": 4
}
```

---

### 2. Get Active Jobs

**Endpoint:** `GET /api/v1/scheduler/jobs`

```bash
curl "http://localhost:8000/api/v1/scheduler/jobs"
```

**Response:**
```json
{
  "status": "success",
  "active_jobs": {
    "5m": {
      "display_name": "5 Minutes",
      "interval_minutes": 5,
      "next_run_time": "2025-09-28T22:25:00+03:30",
      "job_id": "volume_analysis_5m",
      "status": "running"
    },
    "1h": {
      "display_name": "1 Hour",
      "interval_minutes": 60,
      "next_run_time": "2025-09-28T23:00:00+03:30",
      "job_id": "volume_analysis_1h",
      "status": "running"
    }
  },
  "total_active": 2
}
```

---

### 3. Start Timeframe Job

**Endpoint:** `POST /api/v1/scheduler/jobs/{timeframe}/start`

```bash
# Start 5-minute analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/5m/start"

# Start 1-hour analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/1h/start"

# Start 4-hour analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/4h/start"
```

**Response:**
```json
{
  "status": "success",
  "message": "Volume analysis job started for 5m",
  "timeframe": "5m"
}
```

---

### 4. Stop Timeframe Job

**Endpoint:** `POST /api/v1/scheduler/jobs/{timeframe}/stop`

```bash
# Stop 5-minute analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/5m/stop"

# Stop 1-hour analysis
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/1h/stop"
```

**Response:**
```json
{
  "status": "success",
  "message": "Volume analysis job stopped for 5m",
  "timeframe": "5m"
}
```

---

### 5. Stop All Jobs

**Endpoint:** `POST /api/v1/scheduler/jobs/stop-all`

```bash
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/stop-all"
```

**Response:**
```json
{
  "status": "success",
  "message": "All volume analysis jobs stopped successfully"
}
```

## 🎯 Common Use Cases

### 1. Monitor All Active Jobs

```bash
# Check what's currently running
curl "http://localhost:8000/api/v1/scheduler/jobs" | jq '.active_jobs'
```

### 2. Start Multiple Timeframes

```bash
# Start short-term monitoring
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/5m/start"

# Start medium-term monitoring  
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/1h/start"
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/4h/start"

# Start long-term monitoring
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/1d/start"
```

### 3. Stop Specific Timeframes

```bash
# Stop only short-term monitoring
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/5m/stop"

# Keep 1h, 4h and 1d running
```

### 4. Emergency Stop All

```bash
# Stop everything immediately
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/stop-all"
```

### 5. Check Available Options

```bash
# See all available timeframes
curl "http://localhost:8000/api/v1/scheduler/timeframes" | jq '.available_timeframes'
```

## 📊 Monitoring Examples

### Shell Script for Job Management

```bash
#!/bin/bash

API_BASE="http://localhost:8000/api/v1"

# Function to show all active jobs
show_jobs() {
    echo "📊 Active Jobs:"
    curl -s "$API_BASE/scheduler/jobs" | jq -r '.active_jobs | to_entries[] | "\(.key): \(.value.display_name) - Next: \(.value.next_run_time)"'
}

# Function to start multiple timeframes
start_monitoring() {
    echo "🚀 Starting monitoring jobs..."
    curl -s -X POST "$API_BASE/scheduler/jobs/5m/start" | jq '.message'
    curl -s -X POST "$API_BASE/scheduler/jobs/1h/start" | jq '.message'
    curl -s -X POST "$API_BASE/scheduler/jobs/4h/start" | jq '.message'
}

# Function to stop all jobs
stop_all() {
    echo "🛑 Stopping all jobs..."
    curl -s -X POST "$API_BASE/scheduler/jobs/stop-all" | jq '.message'
}

# Menu
case "$1" in
    status|show)
        show_jobs
        ;;
    start)
        start_monitoring
        ;;
    stop)
        stop_all
        ;;
    *)
        echo "Usage: $0 {status|start|stop}"
        exit 1
        ;;
esac
```

### Python Management Script

```python
import httpx
import asyncio
import json

class VolumeSchedulerManager:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient()
    
    async def get_active_jobs(self):
        """Get all active jobs."""
        response = await self.client.get(f"{self.base_url}/api/v1/scheduler/jobs")
        return response.json()
    
    async def start_timeframe(self, timeframe):
        """Start a specific timeframe."""
        response = await self.client.post(f"{self.base_url}/api/v1/scheduler/jobs/{timeframe}/start")
        return response.json()
    
    async def stop_timeframe(self, timeframe):
        """Stop a specific timeframe."""
        response = await self.client.post(f"{self.base_url}/api/v1/scheduler/jobs/{timeframe}/stop")
        return response.json()
    
    async def stop_all(self):
        """Stop all jobs."""
        response = await self.client.post(f"{self.base_url}/api/v1/scheduler/jobs/stop-all")
        return response.json()
    
    async def get_available_timeframes(self):
        """Get available timeframes."""
        response = await self.client.get(f"{self.base_url}/api/v1/scheduler/timeframes")
        return response.json()

# Usage example
async def main():
    manager = VolumeSchedulerManager()
    
    # Show available timeframes
    timeframes = await manager.get_available_timeframes()
    print("Available timeframes:", list(timeframes["available_timeframes"].keys()))
    
    # Start some jobs
    await manager.start_timeframe("5m")
    await manager.start_timeframe("1h")
    
    # Check status
    jobs = await manager.get_active_jobs()
    print(f"Active jobs: {len(jobs['active_jobs'])}")
    
    # Stop all
    await manager.stop_all()

# Run: asyncio.run(main())
```

## 🔄 Migration from Old API

### Old Way (Single Job)
```bash
# Old API - only one job at a time
curl -X POST "http://localhost:8000/api/v1/scheduler/start"
curl -X POST "http://localhost:8000/api/v1/scheduler/stop"
```

### New Way (Multi-Timeframe)
```bash
# New API - multiple concurrent jobs
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/5m/start"
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/1h/start"
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/4h/start"

# Stop specific timeframes
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/5m/stop"

# Or stop all
curl -X POST "http://localhost:8000/api/v1/scheduler/jobs/stop-all"
```

## ⚡ Benefits

### Granular Control
- **Independent Schedules**: Run 5min and 1hour analysis simultaneously
- **Selective Stopping**: Stop only specific timeframes
- **Resource Management**: Control analysis frequency per timeframe

### Better Monitoring
- **Real-time Status**: See next run times for each timeframe
- **Job Tracking**: Individual job IDs and status
- **Clear Overview**: Know exactly what's running

### Flexible Operations
- **Mix Timeframes**: Combine short and long-term analysis
- **Emergency Control**: Stop all jobs instantly
- **Easy Management**: Simple REST API for all operations

## 🚨 Important Notes

### Timezone
All schedules run in **Asia/Tehran (IRST)** timezone.

### Concurrency
Each timeframe job runs independently with:
- **Max instances**: 1 per timeframe
- **Coalescing**: Missed runs are combined
- **Misfire grace**: 60 seconds

### Backward Compatibility
Old API endpoints still work but map to the new system:
- `/scheduler/start` → starts `5m` job
- `/scheduler/stop` → stops all jobs

### Resource Usage
Multiple timeframes will increase resource usage. Monitor system performance when running many concurrent jobs.

## 🛠 Troubleshooting

### Common Issues

1. **Job Already Running**
   ```json
   {"detail": "Job for timeframe 5m is already running"}
   ```
   **Solution**: Check active jobs first, or stop existing job

2. **Invalid Timeframe**
   ```json
   {"detail": "Invalid timeframe: 10m. Available: ['5m', '1h', '4h', '1d']"}
   ```
   **Solution**: Use `/scheduler/timeframes` to see valid options

3. **No Active Jobs**
   ```json
   {"active_jobs": {}, "total_active": 0}
   ```
   **Solution**: Start jobs using `/scheduler/jobs/{timeframe}/start`

### Debug Commands

```bash
# Check service health
curl "http://localhost:8000/"

# See all available timeframes
curl "http://localhost:8000/api/v1/scheduler/timeframes"

# Check what's running
curl "http://localhost:8000/api/v1/scheduler/jobs"

# Get detailed scheduler info
curl "http://localhost:8000/api/v1/scheduler/info"
```

This enhanced API gives you complete control over volume analysis scheduling with the granularity and flexibility you requested!
