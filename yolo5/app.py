import time
import uuid
import yaml
import boto3
from pathlib import Path
from flask import Flask, request, jsonify
from detect import run
from loguru import logger
import os
from pymongo import MongoClient

# Set up Boto3 client
s3_client = boto3.client('s3')

# MongoDB connection
mongo_client = MongoClient('mongodb://mongo1:27017,mongo2:27017,mongo3:27017/?replicaSet=myReplicaSet')
db = mongo_client['prediction_database']
predictions_collection = db['predictions']

app = Flask(__name__)
images_bucket = os.environ['S3_BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

if not os.path.exists('temp/photos/'):
    os.makedirs('temp/photos/')


@app.route('/predict', methods=['POST'])
def predict():
    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')

    img_name = request.args.get('imgName')

    # Extract the file name from the full path
    file_name = os.path.basename(img_name)
    original_img_path = f'temp/photos/{file_name}'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(original_img_path), exist_ok=True)

    # Download image from S3
    try:
        s3_client.download_file(images_bucket, img_name, original_img_path)
    except Exception as e:
        logger.error(f'Error downloading the image from S3: {e}')
        return f'Error: {e}', 500

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predict objects in the image
    run(weights='yolov5s.pt', data='data/coco128.yaml', source=original_img_path, project='static/data',
        name=prediction_id, save_txt=True)
    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # Path for the predicted image with labels
    predicted_img_dir = Path(f'static/data/{prediction_id}')
    predicted_img_path = predicted_img_dir / file_name

    # Ensure the predicted image path is correct
    if not predicted_img_path.exists():
        logger.error(f'Predicted image does not exist: {predicted_img_path}')
        return f'Error: Predicted image does not exist', 500

    # Upload predicted image to S3
    try:
        s3_client.upload_file(str(predicted_img_path), images_bucket, f'predictions/{file_name}')
    except Exception as e:
        logger.error(f'Error uploading the predicted image to S3: {e}')
        return f'Error: {e}', 500

    # Parse prediction labels and create a summary
    pred_summary_path = predicted_img_dir / f'labels/{file_name.split(".")[0]}.txt'
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = [line.split(' ') for line in f.read().splitlines()]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path.as_posix(),
            'labels': labels,
            'time': time.time()
        }

        # Store prediction summary in MongoDB
        result = predictions_collection.insert_one(prediction_summary)

        # Retrieve the inserted document and convert _id to string
        inserted_document = predictions_collection.find_one({'_id': result.inserted_id})
        inserted_document['_id'] = str(inserted_document['_id'])

        return jsonify(inserted_document)
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
