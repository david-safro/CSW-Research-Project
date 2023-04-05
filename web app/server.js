const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');
const session = require('express-session');
const fs = require('fs');

// Set the view engine to EJS
app.set('view engine', 'ejs');

// Set the directory for the views
app.set('views', path.join(__dirname, 'public'));

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Use body-parser middleware to parse request bodies
app.use(bodyParser.urlencoded({ extended: false }));

// Use session middleware
app.use(session({
  secret: 'mysecret',
  resave: false,
  saveUninitialized: true
}));

// Read the users data from the users.json file
const usersData = fs.readFileSync('users.json');
const users = JSON.parse(usersData);

// Render the login page when the user visits /login
app.get('/login', (req, res) => {
  res.render('login');
});

// Handle login form submissions
app.post('/login', (req, res) => {
  const username = req.body.username;
  const password = req.body.password;
  
  // Find the user with the matching username and password in the users array
  const user = users.find(u => u.username === username && u.password === password);
  
  if (user) {
    // If the user exists, store their username in the session
    req.session.username = user.username;
    // Redirect to the home page
    res.redirect('/');
  } else {
    // If the user doesn't exist, render the login page with an error message
    res.render('login', { error: 'Invalid username or password' });
  }
});

// Render the home page
app.get('/', (req, res) => {
  // Check if the user is logged in
  if (req.session.username) {
    // If the user is logged in, render the index page with the user's name
    res.render('index', { username: req.session.username });
  } else {
    // If the user is not logged in, redirect to the login page
    res.redirect('/login');
  }
});

// Handle logout
app.get('/logout', (req, res) => {
  // Destroy the session
  req.session.destroy();
  // Redirect to the login page
  res.redirect('/login');
});

// Start the server
app.listen(3000, () => {
  console.log('Server running on port 3000');
});
