# -*- coding: utf-8 -*-
"""
Flask REST API for Trilingual Sentiment Classification
Endpoints for real-time sentiment predictions

Author: Deepika
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline
from langdetect import detect, LangDetectException
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "./sentiment_model_final"
SUPPORTED_LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'te': 'Telugu'}
DEVICE = 0 if torch.cuda.is_available() else -1

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APPLICATION
# ============================================================================
app = Flask(__name__)
CORS(app)

# ============================================================================
# GLOBAL MODEL VARIABLES
# ============================================================================
sentiment_pipeline = None


def load_model():
    """
    Load the fine-tuned sentiment classification model.
    """
    global sentiment_pipeline
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        sentiment_pipeline = pipeline(
            task='text-classification',
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=DEVICE,
            top_k=None  # Return all class scores
        )
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return False


def detect_language(text):
    """
    Detect language of input text.
    Returns language code (en, hi, te) or None if not supported.
    """
    try:
        lang = detect(text)
        if lang in SUPPORTED_LANGUAGES:
            return lang
        else:
            return None
    except LangDetectException:
        return None


def validate_input(text):
    """
    Validate input text.
    """
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"
    
    if len(text.strip()) == 0:
        return False, "Text cannot be empty or whitespace only"
    
    if len(text) > 512:
        return False, "Text is too long (max 512 characters)"
    
    return True, "Valid"


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': sentiment_pipeline is not None,
        'supported_languages': SUPPORTED_LANGUAGES
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for input text.
    
    Request JSON:
    {
        "text": "Your text here",
        "language": "en"  (optional, will auto-detect if not provided)
    }
    
    Response JSON:
    {
        "text": "Your text here",
        "language": "en",
        "language_name": "English",
        "sentiment": "Positive",
        "confidence": 0.95,
        "scores": {
            "Positive": 0.95,
            "Neutral": 0.03,
            "Negative": 0.02
        },
        "status": "success"
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body must be JSON'
            }), 400
        
        text = data.get('text', '').strip()
        language = data.get('language', None)
        
        # Validate input
        is_valid, message = validate_input(text)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': message
            }), 400
        
        # Detect language if not provided
        if not language:
            language = detect_language(text)
            if not language:
                return jsonify({
                    'status': 'error',
                    'message': 'Unsupported language. Supported: English, Hindi, Telugu'
                }), 400
        else:
            if language not in SUPPORTED_LANGUAGES:
                return jsonify({
                    'status': 'error',
                    'message': f'Unsupported language: {language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}'
                }), 400
        
        # Make prediction
        logger.info(f"Predicting sentiment for: {text[:50]}... (lang: {language})")
        predictions = sentiment_pipeline(text)[0]
        
        # Sort by score descending and get top prediction
        predictions_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)
        top_prediction = predictions_sorted[0]
        
        # Format response
        response = {
            'text': text,
            'language': language,
            'language_name': SUPPORTED_LANGUAGES[language],
            'sentiment': top_prediction['label'],
            'confidence': round(top_prediction['score'], 4),
            'scores': {
                pred['label']: round(pred['score'], 4)
                for pred in predictions_sorted
            },
            'status': 'success'
        }
        
        logger.info(f"Prediction: {top_prediction['label']} ({top_prediction['score']:.4f})")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple texts.
    
    Request JSON:
    {
        "texts": [
            "Text 1",
            "Text 2",
            "Text 3"
        ]
    }
    
    Response JSON:
    {
        "predictions": [
            {
                "text": "Text 1",
                "language": "en",
                "sentiment": "Positive",
                "confidence": 0.95
            },
            ...
        ],
        "total": 3,
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body must be JSON'
            }), 400
        
        texts = data.get('texts', [])
        
        if not isinstance(texts, list):
            return jsonify({
                'status': 'error',
                'message': 'texts must be a list'
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                'status': 'error',
                'message': 'texts list cannot be empty'
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'status': 'error',
                'message': 'Maximum 100 texts per request'
            }), 400
        
        predictions = []
        for text in texts:
            text = text.strip()
            is_valid, _ = validate_input(text)
            
            if not is_valid:
                predictions.append({
                    'text': text,
                    'sentiment': None,
                    'confidence': None,
                    'error': 'Invalid input'
                })
                continue
            
            try:
                lang = detect_language(text)
                if not lang:
                    predictions.append({
                        'text': text,
                        'sentiment': None,
                        'confidence': None,
                        'error': 'Unsupported language'
                    })
                    continue
                
                result = sentiment_pipeline(text)[0]
                top = sorted(result, key=lambda x: x['score'], reverse=True)[0]
                predictions.append({
                    'text': text,
                    'language': lang,
                    'sentiment': top['label'],
                    'confidence': round(top['score'], 4)
                })
            except Exception as e:
                predictions.append({
                    'text': text,
                    'sentiment': None,
                    'confidence': None,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'total': len(predictions),
            'status': 'success'
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Batch prediction failed: {str(e)}'
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """
    Get API and model information.
    """
    return jsonify({
        'api_version': '1.0.0',
        'model_name': 'DistilBERT (fine-tuned)',
        'task': 'Trilingual Sentiment Classification',
        'languages': SUPPORTED_LANGUAGES,
        'sentiment_classes': ['Positive', 'Neutral', 'Negative'],
        'model_path': MODEL_PATH,
        'device': 'GPU' if DEVICE == 0 else 'CPU'
    }), 200


@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    """
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found. Available endpoints: /predict, /predict_batch, /health, /info'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """
    Handle 405 errors.
    """
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405


# ============================================================================
# APPLICATION STARTUP
# ============================================================================
@app.before_request
def before_request():
    """
    Initialize model before first request.
    """
    if sentiment_pipeline is None:
        load_model()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TRILINGUAL SENTIMENT CLASSIFIER - FLASK API")
    print("="*70)
    print("\nLoading model...")
    
    if load_model():
        print("\n✅ API Ready!")
        print("\nEndpoints:")
        print("  POST /predict         - Single text prediction")
        print("  POST /predict_batch   - Batch predictions")
        print("  GET  /health          - Health check")
        print("  GET  /info            - API information")
        print("\nStarting Flask server on http://127.0.0.1:5000")
        print("Press CTRL+C to stop\n")
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    else:
        print("\n❌ Failed to load model. Exiting.")
        exit(1)
