FROM python:3.11-slim

WORKDIR /app

# Install dependencies — this file's build context is the repo root (Railway's default
# Dockerfile path is just "Dockerfile" at the service root), so the source path needs the
# core/ prefix; requirements.txt does not exist at the repo root itself.
COPY core/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

# Set working directory to core for imports
WORKDIR /app/core

# postgres.railway.internal only resolves from inside Railway's network, so migrations
# can't be run from a local machine against production — this is the only place they can
# run. Matches the alembic-upgrade-then-serve pattern docker-compose.yml already uses
# locally.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]