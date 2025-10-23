
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -e .
CMD ["python","scripts/run_demo.py","--config","configs/demo.yaml"]
