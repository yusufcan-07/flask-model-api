from flask import Flask, request, jsonify
from werkzeug.exceptions import InternalServerError
from pydantic import BaseModel, ValidationError
from transformers import pipeline

app = Flask(__name__)

class NewsItem(BaseModel):
    content: str

def load_model():
    # Dynamically load the model
    model_name = "savasy/bert-base-turkish-sentiment-cased"
    model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return model

@app.route('/analyze-news/', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        item = NewsItem(**data)
        
        # Load model for each request (consider caching if performance is an issue)
        nlp = load_model()
        result = nlp(item.content)

        # Optional: Clear model from memory if needed (can impact performance)
        # del nlp

        return jsonify(result)

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        raise InternalServerError(description=str(e))

if __name__ == '__main__':
    app.run(debug=True)
