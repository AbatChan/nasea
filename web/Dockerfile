FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY web/requirements.txt web/requirements.txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt

# Copy the rest
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run chainlit
CMD ["chainlit", "run", "web/app.py", "--host", "0.0.0.0", "--port", "8000"]
