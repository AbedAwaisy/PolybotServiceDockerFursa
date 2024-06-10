import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import requests
import boto3
import json
from collections import Counter

class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', certificate=open('/home/ubuntu/bot.pem', 'r') , timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_url):
        super().__init__(token, telegram_chat_url)
        self.s3_bucket_name = s3_bucket_name
        self.yolo5_url = yolo5_url
        self.s3_client = boto3.client('s3')

    def upload_to_s3(self, file_path):
        try:
            self.s3_client.upload_file(file_path, self.s3_bucket_name, file_path)
            return f'{file_path}'
        except Exception as e:
            logger.error(f'Error uploading to S3: {e}')
            raise

    def download_from_s3(self, s3_path, local_path):
        try:
            self.s3_client.download_file(self.s3_bucket_name, s3_path, local_path)
        except Exception as e:
            logger.error(f'Error downloading from S3: {e}')
            raise

    def send_yolo5_request(self, file_path):
        try:
            response = requests.post(f'{self.yolo5_url}/predict', params={'imgName': file_path})

            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error sending request to yolo5: {e}')
            raise

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            self.send_text(msg['chat']['id'], f'Photo received. Processing...')

            try:
                # Upload the photo to S3
                s3_file_path = self.upload_to_s3(photo_path)

                # Send an HTTP request to the `yolo5` service for prediction
                prediction_result = self.send_yolo5_request(s3_file_path)

                # Format the prediction results
                labels = [label['class'] for label in prediction_result['labels']]
                label_counts = Counter(labels)
                pretty_result = json.dumps(label_counts, indent=4)

                # Send the formatted results to the Telegram end-user
                self.send_text(msg['chat']['id'], f'Prediction result:\n{pretty_result}')

                # Download the predicted image from S3
                predicted_img_s3_path = f'predictions/{os.path.basename(photo_path)}'
                local_predicted_img_path = f'temp/predictions/{os.path.basename(photo_path)}'

                # Ensure the local directory exists
                if not os.path.exists('temp/predictions'):
                    os.makedirs('temp/predictions')

                self.download_from_s3(predicted_img_s3_path, local_predicted_img_path)

                # Send the predicted image to the Telegram end-user
                self.send_photo(msg['chat']['id'], local_predicted_img_path)

            except Exception as e:
                self.send_text(msg['chat']['id'], f'Error processing the photo: {e}')
        else:
            self.send_text(msg['chat']['id'], 'Please send a photo for object detection.')
