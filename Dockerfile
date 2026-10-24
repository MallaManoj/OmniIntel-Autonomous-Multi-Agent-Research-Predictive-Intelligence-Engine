# ==============================================================================
# OmniIntel Autonomous Multi-Agent Research & Predictive Intelligence Engine
# Production Multi-Stage Dockerfile (CIS Benchmark & Security Hardened)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Wheel Compilation
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set build-time optimization flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install system build tools for compiling PyTorch C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated production virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies into /opt/venv
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# Set production runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    APP_ENV=production

# Install minimal runtime utilities (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged non-root system user and group (UID 10001) for CIS compliance
RUN groupadd -g 10001 omniintel && \
    useradd -u 10001 -g omniintel -s /bin/bash -m omniintel

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code and configuration with unprivileged ownership
COPY --chown=omniintel:omniintel src/ /app/src/
COPY --chown=omniintel:omniintel configs/ /app/configs/
COPY --chown=omniintel:omniintel scripts/ /app/scripts/
COPY --chown=omniintel:omniintel pyproject.toml /app/pyproject.toml
COPY --chown=omniintel:omniintel README.md /app/README.md

# Ensure proper directory permissions
RUN chown -R omniintel:omniintel /app

# Switch to unprivileged non-root user
USER omniintel

# Expose API port
EXPOSE 8000

# Automated container healthcheck against FastAPI liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server entrypoint with standard proxy headers enabled
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips=*"]
