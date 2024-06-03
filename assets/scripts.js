let slideIndex = 1;
const slides = document.querySelectorAll('.slide');
console.log(slides)

showSlides(slideIndex);

function plusSlides(n) {
    showSlides(slideIndex += n);
    }
    
function showSlides(n) {
    if (n > slides.length) {
        slideIndex = 1;
    }
    if (n < 1) {
        slideIndex = slides.length;
    }
    slides.forEach(slide => {
        slide.classList.remove('active');
    });
    slides[slideIndex - 1].classList.add('active');
}


function uploadData() {
    // Get the form element
    const form = document.getElementById('uploadForm');
    
    // Create a FormData object from the form
    const formData = new FormData(form);

    // Create an XMLHttpRequest object
    const xhr = new XMLHttpRequest();

    // Define the endpoint URL where you want to send the data
    const url = 'upload.php'; // Replace 'upload.php' with your actual server endpoint

    // Set up the request
    xhr.open('POST', url, true);

    // Define the function to handle the response from the server
    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status === 200) {
          // Upload successful
          console.log('Upload successful');
        } else {
          // Upload failed
          console.error('Upload failed');
        }
      }
    };

    // Send the request with the FormData
    xhr.send(formData);
  }
