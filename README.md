### Tweet Sentiment Extraction

https://www.kaggle.com/c/tweet-sentiment-extraction/overview

The main goal for this competition is to create a model to be able to classify a tweet.

### Install dependencies

In order to install the required dependencies, you have to run the following command at root level.

```
pip install -r requirements.txt
```

### Run flask

Run the runFlask.cmd on buildingScripts folder.

Access it at http://127.0.0.1:5000/

### Run Angular Project

Run the runAngular.cmd on buildingScripts folder.

Access it at http://localhost:2500/

In order to run the following train models:

trainOverSamplingBOW.py
trainOverSamplingTfidf.py
trainOverSamplingWord2Vec.py
trainUnderSamplingWord2Vec.py
trainUnderSamplingBOW.py
trainUnderSamplingTfidf.py

Run the tranAllTheModels.cmd on buildingScripts folder.

In the apiService folder you can find the package.

In the evaluateOverTestData you can find all the files to be executed. 
