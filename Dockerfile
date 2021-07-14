# init base image
FROM python:3.8.3
# define the present working dir
WORKDIR /Tweet_Sentiment_Extraction
# copy the contens into the working dir
ADD . /Tweet_Sentiment_Extraction
# run pip to install the dependencies of the app
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install pyresparser && python -m nltk.downloader punkt && python -m nltk.downloader wordnet && python -m nltk.downloader averaged_perceptron_tagger
# RUN cd apiService
WORKDIR /Tweet_Sentiment_Extraction/apiService
ENV FLASK_APP=api.py
CMD ["python", "api.py"]

# docker image build -t docker-flask-app .
# docker run -p 5000:5000 -d docker-flask-app

# docker tag docker-flask-app mixaverross88/docker-flask-app:latest
# docker push mixaverross88/docker-flask-app:latest

