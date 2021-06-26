echo %cd%
cd ../

python trainOverSamplingBOW.py > presentation\results\trainModelsOutput\trainOverSamplingBOW.txt
python trainOverSamplingTfidf.py > presentation\results\trainModelsOutput\trainOverSamplingTfidf.txt
python trainOverSamplingWord2Vec.py > presentation\results\trainModelsOutput\trainOverSamplingWord2Vec.txt
python trainUnderSamplingBOW.py > presentation\results\trainModelsOutput\trainUnderSamplingBOW.txt
python trainUnderSamplingTfidf.py > presentation\results\trainModelsOutput\trainUnderSamplingTfidf.txt
python trainUnderSamplingWord2Vec.py > presentation\results\trainModelsOutput\trainUnderSamplingWord2Vec.txt