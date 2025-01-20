function clearMessages() {
    // Clear the content of the #outcomeMessage div
    document.getElementById('outcomeMessage').textContent = '';

    // Clear the content of the .result-container div
    document.querySelector('.result-container').innerHTML = '';

    // Clear the content of the addMessage div
    document.getElementById('addMessage').innerHTML = '';

    // Clear the images
    clearImages();
}

function clearImages() {
    // Get the current timestamp
    var timestamp = new Date().getTime();
    // Clear the image previews
    document.getElementById('image1Display').src = '';
    document.getElementById('image2Display').src = '';
    document.getElementById('searchImageDisplay').src = ''; // Added this line
    document.getElementById('imageDisplay').src = '';

    // Hide the image tags
    document.getElementById('image1Display').style.display = 'none';
    document.getElementById('image2Display').style.display = 'none';
    document.getElementById('searchImageDisplay').style.display = 'none'; // Added this line
    document.getElementById('imageDisplay').style.display = 'none';

    // Clear the search results
    document.querySelector('.result-container').innerHTML = '';

    // Clear the input fields
    document.getElementById('image1').value = '';
    document.getElementById('image2').value = '';
    document.getElementById('searchImage').value = ''; // Added this line
    document.getElementById('image').value = '';
}

function previewImage(input, displayId) {
    const reader = new FileReader();

    reader.onloadend = function() {
        const img = document.getElementById(displayId);
        img.src = reader.result;
        img.style.display = 'block';  // Show the img tag
    }

    if (input.files[0]) {
        reader.readAsDataURL(input.files[0]);
    } else {
        const img = document.getElementById(displayId);
        img.src = '';
        img.style.display = 'none';  // Hide the img tag
    }
}

document.getElementById('image1').addEventListener('change', function() {
    previewImage(this, 'image1Display');
});

document.getElementById('image2').addEventListener('change', function() {
    previewImage(this, 'image2Display');
});

document.getElementById('image').addEventListener('change', function() {
    previewImage(this, 'imageDisplay');
});

document.getElementById('searchImage').addEventListener('change', function() {
    previewImage(this, 'searchImageDisplay');
});

document.getElementById('compareForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from being submitted normally

    const formData = new FormData(this);  // Create a FormData object from the form

    // Send the form data using AJAX
    fetch('/compare', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())  // Parse the JSON response
    .then(data => {
        // Handle the response data
        if (data.error) {
            // If there's an error, display it
            alert(data.error);
        } else {
            // Otherwise, display the images
            document.getElementById('image1Display').src = data.image1_url;
            document.getElementById('image2Display').src = data.image2_url;

            // Update the content of the #outcomeMessage div with the outcome message
            document.getElementById('outcomeMessage').textContent = data.message;
        }
    })
    .catch(error => {
        // Handle any errors
        console.error('Error:', error);
    });
});

document.getElementById('searchForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from being submitted normally

    const formData = new FormData(this);  // Create a FormData object from the form

    // Send the form data using AJAX
    fetch('/search', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())  // Parse the JSON response
    .then(data => {
        // Handle the response data
        if (data.error) {
            // If there's an error, display it
            alert(data.error);
        } else {
            // Clear the previous results
            document.querySelector('.result-container').innerHTML = '';

            // Create a div to display the result
            var resultDiv = document.createElement('div');
            resultDiv.classList.add('result');

            // Add the image path, match percentage, and message to the div
            resultDiv.innerHTML = 
                '<p>Image Path: ' + data.image_path + '</p>' +
                '<p>Match Percentage: ' + data.match_percentage + '</p>' +
                '<p>Message: ' + data.message + '</p>';

            // Append the result div to the result-container
            document.querySelector('.result-container').appendChild(resultDiv);
        }
    })
    .catch(error => {
        // Handle any errors
        console.error('Error:', error);
    });
});

document.getElementById('addForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from being submitted normally

    const formData = new FormData(this);  // Create a FormData object from the form

    // Send the form data using AJAX
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())  // Parse the JSON response
    .then(data => {
        // Handle the response data
        if (data.error) {
            // If there's an error, display it
            alert(data.error);
        } else {
            // Otherwise, display the image
            document.getElementById('imageDisplay').src = data.image_url;

            // Display the server message in the 'addMessage' div
            document.getElementById('addMessage').innerHTML = data.message;
        }
    })
    .catch(error => {
        // Handle any errors
        console.error('Error:', error);
    });
});