@ECHO OFF
set root=%cd%
call %root%\Scripts\activate.bat
cd %root%\app.py
pip install -r requirements.txt
call streamlit run app.py