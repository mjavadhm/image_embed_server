# --- 1. Use a slim, modern Python base image ---
FROM python:3.9-slim-bookworm

# --- 2. Set the working directory ---
WORKDIR /app

# --- 3. Copy requirements first for caching ---
COPY requirements.txt ./

# --- 4. Install dependencies ---
# gdown is installed separately as it's small
RUN pip install --no-cache-dir gdown && \
    pip install --no-cache-dir -r requirements.txt

# --- 5. Copy your application code ---
COPY ./main.py .

# --- 6. Expose the port ---
EXPOSE 8000

# --- 7. Define the run command ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
