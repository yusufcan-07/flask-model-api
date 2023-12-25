from flask import Flask, request, jsonify
from werkzeug.exceptions import InternalServerError
from pydantic import BaseModel, ValidationError
from transformers import pipeline

app = Flask(__name__)

# Initialize the model (example with a sentiment analysis model)
model_name = "savasy/bert-base-turkish-sentiment-cased"  # Replace with your chosen model
nlp = None

class NewsItem(BaseModel):
    content: str

@app.before_first_request
def load_model():
    global nlp
    nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

@app.route('/analyze-news/', methods=['POST'])
def analyze_news():
    try:
        # Parse and validate input
        data = request.get_json()
        item = NewsItem(**data)

        # Analyze news content
        result = nlp(item.content)
        return jsonify(result)

    except ValidationError as e:
        # Handle pydantic validation error
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        # Handle general errors
        raise InternalServerError(description=str(e))

if __name__ == '__main__':
    app.run(debug=True)
