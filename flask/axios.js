import 'regenerator-runtime/runtime';
import axios from 'axios';

const form = document.querySelector('form');

form.addEventListener('submit', async event => {
  event.preventDefault();

  const formData = new FormData(form);

  var loader = document.querySelector('.loader')
    loader.style.visibility = 'visible';

  await predict(formData);
});


export const predict = async formData => {
  try {
    const response = await axios.post('http://127.0.0.1:5000/predict', formData);

    var loader = document.querySelector('.loader')
      loader.style.visibility = 'hidden';

    const output = response.data;
    console.log('Output', output);

    var split_right = document.querySelector('.right');

    if(output.display_right) {
      split_right.style.display = 'initial';

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