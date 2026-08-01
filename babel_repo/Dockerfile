FROM python:3.12-slim AS builder
COPY . /app
RUN pip install -r /app/requirements.txt

FROM python:3.12-slim
COPY --from=builder /app /app
CMD ["python", "-m", "shop"]
