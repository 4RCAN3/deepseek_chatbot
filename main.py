from flask import Flask, request, jsonify
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map = device)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(device)

# Flask app initialization
app = Flask(__name__)
lock = threading.Lock()  # For handling simultaneous requests safely

@app.route('/process', methods=['POST'])
def process_request():
    """
    Handles incoming requests, passes them to the model for processing,
    and returns the result.
    """
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide a 'text' field in the JSON payload."}), 400

    input_text = data['text']
    print(f'Processings text: {input_text}')
    messages = [{"role": "user", "content": input_text}]

    try:
        # Lock for thread-safe processing
        with lock:
            # Tokenize the input text
            inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([inputs], return_tensors="pt").to(device)
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=100, do_sample=True)
            #Generate output ids
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            
            #Decode the ids to get the response
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.replace('<think>\n\n</think>\n\n', '')
            return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    """
    WebUI for the DeepSeek R1-1.5 Model.
    """
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DeepSeek R1-1.5 Model</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f9;
                color: #333;
            }
            header {
                background-color: #4caf50;
                color: white;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .container {
                max-width: 800px;
                margin: 2rem auto;
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 {
                margin-bottom: 1rem;
            }
            textarea {
                width: 100%;
                padding: 10px;
                margin-bottom: 1rem;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 1rem;
                resize: vertical;
            }
            button {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 1rem;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #45a049;
            }
            pre {
                background: #f4f4f9;
                padding: 1rem;
                border-radius: 5px;
                border: 1px solid #ddd;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin-top: 1rem;
            }
            footer {
                text-align: center;
                margin-top: 2rem;
                font-size: 0.9rem;
                color: #666;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>DeepSeek R1-1.5 Model WebUI</h1>
        </header>
        <div class="container">
            <p>Enter text below to process it using the DeepSeek R1-1.5 model:</p>
            <form action="/process" method="post" onsubmit="return sendRequest(event)">
                <textarea id="text" placeholder="Type your input here..." rows="6"></textarea>
                <button type="submit">Process</button>
            </form>
            <pre id="response"></pre>
        </div>
        <footer>
            <p>Powered by DeepSeek AI | Built with Flask and Hugging Face Transformers</p>
        </footer>
        <script>
            async function sendRequest(event) {
                event.preventDefault();
                const text = document.getElementById('text').value;
                const responseElement = document.getElementById('response');
                if (!text.trim()) {
                    responseElement.textContent = "Please enter some text to process.";
                    return;
                }
                responseElement.textContent = "Processing...";
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });
                const result = await response.json();
                responseElement.textContent = JSON.stringify(result, null, 2);
            }
        </script>
    </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
