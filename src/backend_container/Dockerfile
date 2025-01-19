# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Leverage caching for dependencies 
COPY requirements.txt /app/
COPY /model/ /app/model/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py /app/

# Expose the API port
EXPOSE 5000

CMD ["uvicorn", "main:main", "--host", "0.0.0.0", "--port", "5000"]