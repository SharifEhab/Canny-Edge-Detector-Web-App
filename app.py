from flask import Flask, render_template,request, jsonify, send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def CannyEdgeDetector():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()  # Get the JSON data sent from the client
    sigma = data['sigma']  # Extract sigma value
    image_data = data['image_data']  # Extract image data

    # The image data is likely base64 encoded, so we need to decode it
    image_data = base64.b64decode(image_data.split(',')[1])

    # Convert the image data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))

    
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #LOGIC HEREEE
    
    
    
    
    
    
    
    
    
    
    # Save the processed image to a file
    output_path = os.path.join(os.path.dirname(__file__), 'image.jpg')
    cv2.imwrite(output_path, cv2_image)

    # Return a response
    return send_file('image.jpg', mimetype='image/jpg')
if __name__ == '__main__':
    app.run(debug=True)