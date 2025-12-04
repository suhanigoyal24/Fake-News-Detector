@echo off
REM --- Activate the virtual environment ---
echo Activating virtual environment...
call venv\Scripts\activate

REM --- Run the Streamlit app and open browser automatically ---
echo Running Fake News Detector...
streamlit run app.py --browser.serverAddress=localhost

REM --- Keep the window open after closing ---
pause
