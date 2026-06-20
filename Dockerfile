FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
COPY main.py .
CMD ["python", "main.py"]
