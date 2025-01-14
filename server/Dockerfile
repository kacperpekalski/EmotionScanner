FROM python:3.12-slim

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_front.py --main.port=8501 --main.enableCORS=false"]
