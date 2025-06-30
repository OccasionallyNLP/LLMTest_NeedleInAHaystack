FROM huggingface/transformers-pytorch-gpu:latest

ENV PYTHONPATH /app
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install flash-attn --no-build-isolation