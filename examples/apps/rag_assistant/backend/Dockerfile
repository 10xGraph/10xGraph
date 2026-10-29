# Backend image for the docs assistant. Same layout `10xgraph build` generates:
# two stages, non-root user, gunicorn with uvicorn workers, long graceful drain on SIGTERM.

# ---- Stage 1: install dependencies into a virtualenv ----
# For reproducible production builds, pin both FROM lines by digest
# (python:3.13-slim@sha256:...) and install from a hash-locked
# requirements file (pip install --require-hashes -r requirements.txt).
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Every 10xGraph dependency ships prebuilt wheels, so no compiler is needed. If one
# of your own dependencies has to build from source, uncomment this. It only lives
# in the builder stage and never reaches the final image.
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Install exactly the dependencies listed in your requirements file
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Production process manager: gunicorn supervising uvicorn workers. These are
# not dependencies of 10xgraph-api, so the image installs them here.
RUN pip install "gunicorn>=23,<27" "uvicorn-worker>=0.4,<1"

# Strip CLI-only assets the server never loads: project scaffolds, the bundled
# agent skill, and the local config editor (HTML/JS). Only the server code and
# its dependencies stay in the image.
RUN python -I -c "import importlib.util, pathlib, shutil; spec = importlib.util.find_spec('tenxgraph_api'); cli = pathlib.Path(spec.origin).parent / 'cli' if spec and spec.origin else None; [shutil.rmtree(cli / d, ignore_errors=True) for d in ('templates', 'config_editor') if cli]"

# Remove the CLI's own libraries (typer, rich, questionary and what only they pull
# in). A package that one of your dependencies still needs is kept.
RUN python -I -c "import importlib.util, runpy; importlib.util.find_spec('tenxgraph_api._slim_image') and runpy.run_module('tenxgraph_api._slim_image', run_name='__main__')"

# The server never runs pip, so drop it from the virtualenv the runtime copies
# (about 13 MB). The base image keeps its own pip for debugging.
RUN pip uninstall -y pip

# ---- Stage 2: runtime image ----
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"
# Default to production; overridable via .env or runtime env.
# In production the local in-memory telemetry store stays disabled
# (use OTEL / a publisher for observability instead).
ENV MODE=production
# Gunicorn worker count. Gunicorn reads WEB_CONCURRENCY natively; a sane default
# here beats gunicorn's built-in default of 1 worker. Tune to your CPU/memory
# at deploy time, e.g. `docker run -e WEB_CONCURRENCY=8 ...`.
ENV WEB_CONCURRENCY=2

# Create a non-root user with a fixed uid that owns the work directory.
# uploads/ exists up front so a volume mounted there inherits that owner.
RUN groupadd -r -g 10001 appuser \
    && useradd -r -u 10001 -g appuser -d /app -s /usr/sbin/nologin appuser \
    && mkdir -p /app/uploads && chown appuser:appuser /app /app/uploads
WORKDIR /app

# Only the installed virtualenv comes over from the builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code. --chown sets ownership while copying; a separate
# `chown -R` afterwards would write a second copy of every file into a new layer.
COPY --chown=appuser:appuser . .
# Numeric, so Kubernetes runAsNonRoot can verify it
USER 10001:10001

# Expose port
EXPOSE 8000

# Health check (Python stdlib, so the image needs no curl)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ping', timeout=5)" || exit 1

# Run the application (production)
# Use Gunicorn with Uvicorn workers for better performance and multi-core
# utilization
#
# --graceful-timeout is critical: an agent run can legitimately take
# minutes (the LLM client alone allows up to 600s). Gunicorn's default
# graceful timeout is 30s, so on every rolling deploy SIGTERM would
# hard-kill runs that were still mid-flight, truncating them. Give
# in-flight runs time to finish draining instead.
CMD ["gunicorn", "-k", "uvicorn_worker.UvicornWorker", "-b", "0.0.0.0:8000", "--graceful-timeout", "600", "--timeout", "660", "tenxgraph_api.src.app.main:app"]

