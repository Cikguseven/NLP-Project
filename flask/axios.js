import 'regenerator-runtime/runtime';
import axios from 'axios';

const form = document.querySelector('form');

form.addEventListener('submit', async event => {
  event.preventDefault();

  const formData = new FormData(form);

  var error = document.querySelector('#error');
  error.style.display = 'none';

  var loader = document.querySelector('.loader');
  loader.style.visibility = 'visible';

  var split_right = document.querySelector('.right');
  split_right.style.visibility = 'hidden';

  await predict(formData);
});


export const predict = async formData => {
  try {
    const response = await axios.post('http://127.0.0.1:5000/predict', formData);

    var loader = document.querySelector('.loader')
    loader.style.visibility = 'hidden';

    const output = response.data;

    if (output.error) {
      var error = document.querySelector('#error');
      error.style.display = 'initial';
    }

    else {
      var split_right = document.querySelector('.right');
      split_right.style.visibility = 'visible';

      var echo = document.querySelector('.echo');
      echo.innerHTML = output.tagged_input;

      var offensive_answer = document.querySelector('#answer_1');
      offensive_answer.innerHTML = output.offensive;

      var targeted_answer = document.querySelector('#answer_2');
      targeted_answer.innerHTML = output.targeted;

      var target_answer = document.querySelector('#answer_3');
      target_answer.innerHTML = output.target;
    }

  } catch (errors) {
    console.error(errors);
  }
};