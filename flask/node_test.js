const express = require('express')
const request = require('request');

const app = express();
const PORT = 3000;

app.get('/', function(req, res) {
    res.sendFile('C:/Users/super/Documents/NLP-Project-NEW/flask/template/home_node.html');
    // request('http://127.0.0.1:5000', function (error, response, body) {
    //     console.error('error:', error); // Print the error
    //     console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
    //     console.log('body:', body); // Print the data received
    //     res.send(body); //Display the response on the website
    //   });      
});

app.listen(PORT, function (){ 
    console.log('Listening on Port 3000');
});  