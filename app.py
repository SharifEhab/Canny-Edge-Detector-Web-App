# Import necessary libraries
from flask import Flask, render_template, request, send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np
import os
import cannyfuncyions as functions  # Import custom functions from cannyfuncyions module

# Initialize Flask app
app = Flask(__name__)

# Define route for the main page
@app.route('/')
def CannyEdgeDetector():
    """
    Function to render the main.html template when accessing the root URL.

    Returns:
    - HTML template: The main.html template.
    """
    return render_template('main.html')

# Define route for image upload
@app.route('/upload', methods=['POST'])
def upload():
    """
    Function to handle image upload, processing, and response.

    Returns:
    - File: Processed image file in PNG format.
    """
    # Get JSON data from the request
    data = request.get_json()
    # Extract sigma value for GaussianBlur
    sigma = data['sigma']
    # Extract image data from JSON
    image_data = data['image_data']
    # Extract high and low thresholds for Canny edge detection
    high_threshold_ratio = data['high']
    low_threshold_ratio = data['low']
    # Decode and process image data
    image_data = base64.b64decode(image_data.split(',')[1])
    # Convert image data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))

    # Convert RGB image to BGR format
    cv2_image = functions.convert_rgb_to_bgr(image)

    # Convert BGR image to grayscale
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    # Smooth image using Gaussian blur
    img_smoothed = functions.gaussian_kernel(cv2_image, 10, sigma)

    # Compute image gradient using Sobel operator
    gradientMat, thetaMat = functions.sobel_filters(img_smoothed)

    # Compute non-maximum suppression along the gradient direction
    nonMaxImg = functions.non_max_suppression(gradientMat, thetaMat)

    # Thresholding
    thresholdImg = functions.threshold(nonMaxImg, low_threshold_ratio, high_threshold_ratio)

    # Hysteresis
    img_final = functions.hysteresis(thresholdImg)

    # Save processed image to a file
    output_path = os.path.join(os.path.dirname(__file__), 'image.png')
    cv2.imwrite(output_path, img_final)

    # Return the processed image file
    return send_file('image.png', mimetype='image/png')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
