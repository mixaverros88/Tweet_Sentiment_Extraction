echo %cd%
cd ../
cd api
set FLASK_APP=api.py
rem flask run
python -m flask run