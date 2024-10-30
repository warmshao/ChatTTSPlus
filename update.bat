@echo off
git fetch origin
git reset --hard origin/master
"venv\Scripts\pip.exe" install -r requirements.txt
pause