from flask import Flask, request, render_template_string, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity     
import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import json

# Load and preporcess the data
data = pd.read_json('faq_python.json')

def decode_unicode_escape(text):
    return text.encode('latin1').decode('unicode-escape').encode('latin1').decode('utf-8')

def clean_text(text):
    text = text.replace('â€™', '\'')
    return text

def lowercase_text(text):
    return text.lower()

nltk.download('punkt')
def tokenize_text(text):
    return word_tokenize(text)

# def remove_punctuations(tokens):
#     return [token for token in tokens if token not in string.punctuation]

# nltk.download('stopwords')
# def remove_stopwords(tokens):
#     stop_words = set(stopwords.words('english'))
#     return [token for token in tokens if token not in stop_words]

nltk.download('wordnet')
nltk.download('omw-1.4')
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    text = decode_unicode_escape(text)
    text = clean_text(text)
    text = lowercase_text(text)
    tokens = tokenize_text(text)
    # tokens = remove_punctuations(tokens)
    # tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

data['question_processed'] = data['title'].apply(preprocess_text)

def update_dataset(new_data, file_path='faq_python.json'):
    try:
        with open(file_path, 'r+', encoding='utf-8') as file:
            data = json.load(file)
            data.append(new_data)
            file.seek(0)
            json.dump(data, file, indent=4)

    except FileNotFoundError:
        print("The JSON file was not found.")

# load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

question_embeddings = model.encode(data['question_processed'].tolist())

app = Flask(__name__)

HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chatbot</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <style>
        /* Add your CSS styling for chat interface here */
    </style>
</head>
<body>
    <div id="chat-container">
        <!-- Chat messages will go here -->
    </div>
    <input type="text" id="user-input" placeholder="Ask a question...">
    <button onclick="sendMessage()">Send</button>

    <script>
        function sendMessage() {
            var userQuestion = $('#user-input').val();
            $.ajax({
                url: '/get-answer',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ 'question': userQuestion }),
                success: function(response) {
                    $('#chat-container').append('<div>User: ' + userQuestion + '</div>');
                    $('#chat-container').append('<div>Bot: ' + response.answer + '</div>');
                    $('#user-input').val(''); // Clear input field
                }
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_FORM)

@app.route('/get-answer', methods=['POST'])
def get_answer():
    user_question = request.json['question']
    preprocessed_user_question = preprocess_text(user_question)
    user_question_embedding = model.encode([preprocessed_user_question])
    similarities = cosine_similarity(user_question_embedding, question_embeddings)
    best_match_index = similarities.argmax()
    best_answer = data.iloc[best_match_index]['content']

    new_qa_pair = {'title': user_question, 'content': best_answer}
    update_dataset(new_qa_pair)

    return jsonify({'answer': best_answer})


if __name__ == '__main__':
    app.run(debug=True)