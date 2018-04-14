# Initializes the web site, running the flask server, installing required packages.
# .\venv\Scripts\Activate.ps1

# pip install -r requirements.txt

$Env:FLASK_DEBUG=1
$Env:FLASK_APP= "$($PSScriptRoot)\web\server.py"

python -m flask run