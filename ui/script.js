const inputBox = document.getElementById('inputBox');
const suggestionBox = document.getElementById('suggestionBox');

inputBox.addEventListener('input', function() {
    const inputValue = this.value.trim();
    if (inputValue !== '') {
        fetchMovieSuggestions(inputValue);
    } else {
        clearSuggestions();
    }
});

function fetchMovieSuggestions(inputValue) {
    // Send AJAX request to server-side endpoint to get movie suggestions
    // Replace this with your server-side endpoint URL
    fetch(`http://127.0.0.1:5000/suggestions?query=${inputValue}`)
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        showSuggestions(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function showSuggestions(suggestions) {
suggestionBox.innerHTML = ''; // Clear previous suggestions

const ul = document.createElement('ul'); // Create <ul> element
ul.className = 'suggestion-list'; // Apply the class to the <ul>

suggestions.forEach(suggestion => {
const li = document.createElement('li'); // Create <li> element
li.textContent = suggestion;
li.addEventListener('click', function() {
    inputBox.value = suggestion;
    clearSuggestions();
});
ul.appendChild(li); // Append <li> to <ul>
});

suggestionBox.appendChild(ul); // Append <ul> to suggestionBox
}

function clearSuggestions() {
    suggestionBox.innerHTML = '';
}


document.getElementById('searchBtn').addEventListener('click', function() {
    const movieName = document.getElementById('inputBox').value.trim();
    if (movieName !== '') {
        fetchMoviePosters(movieName);
    } else {
        alert('Please enter a movie name');
    }
});

function fetchMoviePosters(movieName) {
    const formData = new FormData();
    formData.append('movie_name', movieName);

    fetch('http://127.0.0.1:5000/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.length > 0) {
            const container = document.getElementById('moviePosterContainer');
            container.innerHTML = ''; // Clear previous results
            data.forEach(movie => {
                const posterDiv = document.createElement('div');
                posterDiv.className = 'poster';
                const posterImg = document.createElement('img');
                posterImg.src = movie.poster_path;
                const posterName = document.createElement('p');
                posterName.textContent = movie.movie_name;
                
                // Add event listener to display movie details when clicked
                posterDiv.addEventListener('click', function() {
                    showMovieDetails(movie);
                });
                
                posterDiv.appendChild(posterImg);
                posterDiv.appendChild(posterName);
                container.appendChild(posterDiv);
            });
        } else {
            alert('No movies found!');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to fetch movie posters');
    });
}

function showMovieDetails(movie) {
    const modal = document.getElementById('modal');
    const modalPoster = document.querySelector('.modal-poster');
    const modalDetails = document.querySelector('.modal-details');

    // Insert poster
    modalPoster.innerHTML = `<img src="${movie.poster_path}" alt="Movie Poster">`;

    // Insert movie name and description
    modalDetails.innerHTML = `
        <h2>${movie.movie_name}</h2>
        <p>${movie.description}</p>
    `;

    // Display the modal
    modal.style.display = 'block';
}

function closeModal() {
    const modal = document.getElementById('modal');
    modal.style.display = 'none';
}


