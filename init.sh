#!/bin/sh
# Init script for Linux users

pip3 install -r requirements.txt

export FLASK_DEBUG=1
export FLASK_APP=web/server

flask run