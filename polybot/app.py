import flask
from flask import request
import os
from bot import ObjectDetectionBot

app = flask.Flask(__name__)

# Read the TELEGRAM_TOKEN from the Docker secret file
with open('/run/secrets/telegram_dev_token', 'r') as file:
    TELEGRAM_TOKEN = file.read().strip()
TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
YOLO5_URL = os.environ['YOLO5_URL']

@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, YOLO5_URL)

    app.run(host='0.0.0.0', ssl_context=('/home/ubuntu/bot.pem', '/home/ubuntu/bot.key'), port=8443)