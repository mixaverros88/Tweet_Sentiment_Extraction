FROM python:3.8.3
ADD . /Tweet_Sentiment_Extraction
WORKDIR /Tweet_Sentiment_Extraction
ENV FLASK_APP /api/api.py
RUN pip install -r requirements.txt