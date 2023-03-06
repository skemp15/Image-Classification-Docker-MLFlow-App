from flask import Flask, request 
import logging 
from waitress import serve

import mnist_classification as mc

logging.basicConfig(
    format='[%(levelname)s %(name)s] %(asctime)s - %(message)s',
    level = logging.INFO,
    datefmt='%Y/%m/%d %I:%M:%S %p'
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/")
def hello():
    logger.info('Access to landing pge')
    """
    Landing page for image classification
    """
    return('Hello this is the landing page for the MNIST classifier')

@app.route('/classify_images', methods=['POST'])
def classify_images():
    logger.info('Access to classify images')
    json_data = request.get_json()
    response = mc.classify_images(json_data)
    return response

serve(app, port=5030, host='0.0.0.0')
