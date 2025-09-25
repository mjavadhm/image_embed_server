# --- 1. Use a newer, slim, and official Python base image ---
# "bookworm" has a newer version of sqlite3 required by chromadb
FROM python:3.9-slim-bookworm

# --- 2. Set the working directory inside the container ---
WORKDIR /app

# --- 3. Copy ONLY the requirements file first ---
# This is the most important step for using the cache
COPY requirements.txt ./

# --- 4. Install all dependencies in a single layer ---
# The main change is here: explicitly installing torch, torchvision, and torchaudio
# from the CPU-only repository before installing other requirements.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- 5. Copy your application code LAST ---
# Since the application code changes more often than the requirements, we copy it at the end
# so that its changes do not invalidate the package installation layer.
COPY ./main.py .

# --- 6. Expose the port the app runs on ---
EXPOSE 8000

# --- 7. Define the command to run your application ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
