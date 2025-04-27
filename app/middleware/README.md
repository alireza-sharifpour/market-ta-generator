# IP Whitelist Middleware

This middleware restricts access to the API based on client IP addresses. It allows only requests from a predefined list of whitelisted IP addresses.

## Configuration

The IP whitelist is configurable through environment variables:

- `WHITELIST_ENABLED`: Set to `"True"` or `"False"` to enable or disable the whitelist feature. Default is `"True"`.
- `WHITELISTED_IPS`: A comma-separated list of IP addresses allowed to access the API.
  Default: `"127.0.0.1,154.90.55.18,10.72.24.67"`.

Example .env file:

```
WHITELIST_ENABLED=True
WHITELISTED_IPS=127.0.0.1,154.90.55.18,10.72.24.67,192.168.1.100
```

## How It Works

1. The middleware intercepts all incoming HTTP requests.
2. If whitelisting is enabled, it checks if the client's IP address is in the whitelist.
3. If the IP is not in the whitelist, it returns a 403 Forbidden response.
4. If the IP is in the whitelist, the request is passed to the next handler in the chain.

## Implementation

The middleware is implemented as a Starlette BaseHTTPMiddleware class and added to the FastAPI application during startup.

To view the implementation, see [ip_whitelist.py](./ip_whitelist.py).

```python
from your_package.middleware.ip_whitelist import add_ip_whitelist_middleware

app = FastAPI()
add_ip_whitelist_middleware(app)
```
