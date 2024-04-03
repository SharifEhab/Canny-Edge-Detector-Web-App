from flask import Flask, render_template,request, jsonify, send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np
import os


def gaussian_kernel(size, sigma=None):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    if sigma is None:
        sigma = 1.0
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters( img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = cv2.filter2D(img, -1, Kx) #convolution with sobel kernels
    Iy = cv2.filter2D(img, -1, Ky)

    G = np.hypot(Ix, Iy) #magnitude
    G = G / G.max() * 255  #normalize
    theta = np.arctan2(Iy, Ix)  #direction
    return (G, theta)


def non_max_suppression( img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0


            except IndexError as e:
                pass

    return Z

def hysteresis( img):

    M, N = img.shape
    weak = 75
    strong = 255

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img
def threshold(img, lowThreshold, highThreshold):
    print(lowThreshold, highThreshold)

    if highThreshold is not None:
        highThreshold = np.max(img) * highThreshold
    if lowThreshold is not None:
        lowThreshold = highThreshold * lowThreshold

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(75)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res)
    
    
app = Flask(__name__)

@app.route('/')
def CannyEdgeDetector():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()  
    sigma = data['sigma']  # Extract sigma value for GaussianBlur
    image_data = data['image_data']  # Extract image data
    high_threshold = data['high']
    low_threshold = data['low']
    image_data = base64.b64decode(image_data.split(',')[1])
    print(sigma, high_threshold, low_threshold)
    # Convert the image data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))

    
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #LOGIC HEREEE
    
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    #smooth Image using GaussianBlur
    img_smoothed = cv2.filter2D(cv2_image, -1, gaussian_kernel(10, sigma)) #convolution with gaussian kernel
    
    
    #Compute Image gradient using Sobel along with mag and direction
    
    gradientMat, thetaMat = sobel_filters(img_smoothed)
    
    #compute non max supression along the gradient direction at each pixel
    
    nonMaxImg = non_max_suppression(gradientMat, thetaMat)
    
    #Thresholding   
    
    thresholdImg = threshold(nonMaxImg,low_threshold, high_threshold)
    #hesterisis
    img_final = hysteresis(thresholdImg)
    
    
    # Save the processed image to a file
    output_path = os.path.join(os.path.dirname(__file__), 'image.png')
    cv2.imwrite(output_path, img_final)

    # Return a response
    return send_file('image.png', mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=True)