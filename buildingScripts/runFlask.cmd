echo %cd%
cd ../
cd apiService
set FLASK_APP=api.py
rem flask run
python -m flask run