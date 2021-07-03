echo %cd%
cd ../
cd evaluateOverTestData

python evaluateOverSamplingBOW.py > ../presentation\results\evaluateModels\evaluateOverSamplingBOW.txt
python evaluateOverSamplingTFIDF.py > ../presentation\results\evaluateModels\evaluateOverSamplingTFIDF.txt