# Base image: Lightweight Python 3.100 (slim version)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements.txt first (Docker layer caching optimization)
# If requirements.txt doesn't change, dependencies won't reinstall
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port used by Streamlit
EXPOSE 8501

# Command to start the application using Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "server.port=8501"]