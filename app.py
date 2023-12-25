from flask import Flask, request, jsonify
from werkzeug.exceptions import InternalServerError
from pydantic import BaseModel, ValidationError
from transformers import pipeline
from functools import lru_cache
app = Flask(__name__)

class NewsItem(BaseModel):
    content: str
@lru_cache(maxsize=100)
def load_model(model_name):
    # Dynamically load the model
    model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return model

model_name = "savasy/bert-base-turkish-sentiment-cased"
nlp = load_model(model_name)

@app.route('/analyze-news/', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        item = NewsItem(**data)
        
        
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
