const img1 = document.getElementById('img1');
var div1 = document.getElementById('originalimg1');


var div2=document.getElementById('modified');
var btn = document.getElementById('submitBtn');

var inputField = document.getElementById('sigma');
var sigma = parseFloat(inputField.value);

var inputField1 = document.getElementById('low');
var low = parseFloat(inputField1.value);

var inputField2 = document.getElementById('high');
var high = parseFloat(inputField2.value);
//div2.setAttribute('src', '');


img1.addEventListener('change', function (event) {
    // Get the selected file
    div1.setAttribute('src', '');
    var isImageLoaded = false; // Flag to indicate if the image has finished loading

    var file = event.target.files[0];
    var img = new Image();
    img.onload = function() {
        size_img1x = img.naturalWidth;
        size_img1y = img.naturalHeight;
        isImageLoaded = true; // Set the flag to true when the image has finished loading
        handleImageDimensions(size_img1x, size_img1y); 
    }

    img.src = URL.createObjectURL(file);
    function handleImageDimensions(width, height) {
        // You can perform additional operations with the dimensions here
        x1=width;
        y1=height;
        
    }

    // Check if the image has finished loading and the dimensions are available
    if (isImageLoaded) {
        handleImageDimensions(size_img1x, size_img1y);
    } else {
        console.log('Image is still loading...');
    }

    // Read the file as a data URL
    var reader = new FileReader();
    reader.onload = function (event) {
        div1.src = event.target.result; // Set the source of div1 to the data URL
       

        var uploaded_image = reader.result;
    };
    reader.readAsDataURL(file);
});
  
  
  
btn.addEventListener('click', function() {
    // Assuming 'sigma' and 'div1.src' (base64 image data) are defined
    sigma = parseFloat(inputField.value);
    high = parseFloat(inputField2.value);
    low = parseFloat(inputField1.value);
fetch('/upload', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ sigma: sigma, image_data: div1.src, high:high ,low:low })
})
.then(response => response.blob())
.then(image => {
    // Create a local URL for the image
    var image_url = URL.createObjectURL(image);

    // Set the source of an image element to the local URL
    div2.src = image_url;
})
.catch(error => console.error('Error:', error));
   
    ;});