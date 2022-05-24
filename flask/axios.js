import 'regenerator-runtime/runtime';
import axios from 'axios';

function submit_text() {
        if(document.getElementById("textbox").value === "") { 
          document.getElementById("text_submit").disabled = true; 
          }
        else { 
          document.getElementById("text_submit").disabled = false;
          }
        }

const form = document.querySelector('form');

form.addEventListener('submit', async event => {
  event.preventDefault();

  const formData = new FormData(form)

  for (var pair of formData.entries()) {
    console.log(pair[0] + ': ' + pair[1]);
}

  const submitTodoItem = await addTodoItem(formData);
});


export const addTodoItem = async formData => {
  try {
    const response = await axios.post('http://127.0.0.1:5000/predict', formData);
    const newTodoItem = response.data;

    console.log(`Added a new Todo!`, newTodoItem);

    return newTodoItem;
  } catch (errors) {
    console.error(errors);
  }
};