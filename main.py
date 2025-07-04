from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

# Load model lokal (contoh: distilgpt2, ringan)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')

    if not user_input:
        return jsonify({'response': 'Tidak ada input'}), 400

    result = generator(user_input, max_length=100, num_return_sequences=1)
    response_text = result[0]['generated_text']

    return jsonify({'response': response_text})

@app.route('/')
def home():
    return "Wati GPT Bridge is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
