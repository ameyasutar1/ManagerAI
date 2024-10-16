// Smooth scrolling to sections
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});

// Event listeners for buttons
document.getElementById('download-mac-btn').addEventListener('click', function() {
    alert('Download for Mac is not available right now.');
});

document.getElementById('get-chrome-btn').addEventListener('click', function() {
    alert('Get Chrome Extension is under development.');
});
