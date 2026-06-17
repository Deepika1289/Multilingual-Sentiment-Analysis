# -*- coding: utf-8 -*-
"""
Standalone Prediction Script for Sentiment Classification
Useful for testing without Flask API

Author: Deepika
"""

import torch
from transformers import pipeline
from langdetect import detect, LangDetectException
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "./sentiment_model_final"
SUPPORTED_LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'te': 'Telugu'}
DEVICE = 0 if torch.cuda.is_available() else -1

# ============================================================================
# LANGUAGE DETECTION
# ============================================================================
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


# ============================================================================
# SENTIMENT PREDICTION
# ============================================================================
def predict_sentiment(text, model_path=MODEL_PATH):
    """
    Predict sentiment for a given text.
    
    Args:
        text: Input text string
        model_path: Path to the fine-tuned model
    
    Returns:
        Dictionary with prediction results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    sentiment_pipeline = pipeline(
        task='text-classification',
        model=model_path,
        tokenizer=model_path,
        device=DEVICE,
        top_k=None
    )
    
    # Detect language
    language = detect_language(text)
    if not language:
        return {
            'error': 'Unsupported language. Supported: English, Hindi, Telugu',
            'text': text
        }
    
    # Make prediction
    print(f"\nPredicting sentiment for: {text[:60]}...")
    predictions = sentiment_pipeline(text)[0]
    predictions_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)
    top_prediction = predictions_sorted[0]
    
    # Format result
    result = {
        'text': text,
        'language': language,
        'language_name': SUPPORTED_LANGUAGES[language],
        'sentiment': top_prediction['label'],
        'confidence': round(top_prediction['score'], 4),
        'scores': {
            pred['label']: round(pred['score'], 4)
            for pred in predictions_sorted
        }
    }
    
    return result


# ============================================================================
# INTERACTIVE PREDICTION
# ============================================================================
def interactive_mode():
    """
    Interactive prediction mode for testing.
    """
    print("\n" + "="*70)
    print("TRILINGUAL SENTIMENT CLASSIFIER - INTERACTIVE MODE")
    print("="*70)
    print("\nLoading model...")
    
    sentiment_pipeline = pipeline(
        task='text-classification',
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=DEVICE,
        top_k=None
    )
    
    print("✅ Model loaded!")
    print("\nEnter text to analyze sentiment (type 'quit' to exit):\n")
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text.\n")
            continue
        
        # Detect language
        language = detect_language(text)
        if not language:
            print("❌ Unsupported language. Supported: English, Hindi, Telugu\n")
            continue
        
        # Predict
        predictions = sentiment_pipeline(text)[0]
        predictions_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)
        top = predictions_sorted[0]
        
        # Display results
        print(f"\n📍 Language   : {SUPPORTED_LANGUAGES[language]}")
        print(f"😊 Sentiment  : {top['label']}")
        print(f"📊 Confidence : {top['score']*100:.2f}%")
        print(f"\n   Detailed scores:")
        for pred in predictions_sorted:
            print(f"     {pred['label']:10s}: {pred['score']*100:6.2f}%")
        print()


# ============================================================================
# BATCH PREDICTION
# ============================================================================
def batch_predict(texts, model_path=MODEL_PATH):
    """
    Predict sentiment for multiple texts.
    
    Args:
        texts: List of text strings
        model_path: Path to the fine-tuned model
    
    Returns:
        List of prediction results
    """
    print(f"Loading model from {model_path}...")
    sentiment_pipeline = pipeline(
        task='text-classification',
        model=model_path,
        tokenizer=model_path,
        device=DEVICE,
        top_k=None
    )
    
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] Processing: {text[:50]}...")
        
        language = detect_language(text)
        if not language:
            results.append({
                'text': text,
                'error': 'Unsupported language'
            })
            continue
        
        predictions = sentiment_pipeline(text)[0]
        top = sorted(predictions, key=lambda x: x['score'], reverse=True)[0]
        
        results.append({
            'text': text,
            'language': language,
            'sentiment': top['label'],
            'confidence': round(top['score'], 4)
        })
    
    return results


# ============================================================================
# EXAMPLES
# ============================================================================
def run_examples():
    """
    Run example predictions in English, Hindi, and Telugu.
    """
    print("\n" + "="*70)
    print("TRILINGUAL SENTIMENT CLASSIFIER - EXAMPLE PREDICTIONS")
    print("="*70)
    
    examples = [
        ("en", "This product is absolutely amazing! I love it!"),
        ("en", "The delivery was okay. Nothing special."),
        ("en", "Terrible quality. Complete waste of money!"),
        ("hi", "बहुत बढ़िया उत्पाद है! बहुत खुश हूँ।"),
        ("hi", "ठीक है, काम चलाऊ है।"),
        ("hi", "बहुत खराब है! बिल्कुल पसंद नहीं आया।"),
        ("te", "ఈ ఉత్పత్తి అద్భుతమైనది! నేను దీన్ని బాగా సిఫార్సు చేస్తాను!"),
        ("te", "సరిగానే ఉంది, పనిచేస్తుంది."),
        ("te", "చాలా చెడ్డ ఉంది! చిట్టచివర్లకి నిరాశ చెందాను."),
    ]
    
    print("\nLoading model...")
    sentiment_pipeline = pipeline(
        task='text-classification',
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=DEVICE,
        top_k=None
    )
    print("✅ Model loaded!\n")
    
    print(f"{'Language':<10} {'Text':<50} {'Sentiment':<10} {'Confidence'}")
    print("-" * 85)
    
    for lang, text in examples:
        predictions = sentiment_pipeline(text)[0]
        top = sorted(predictions, key=lambda x: x['score'], reverse=True)[0]
        lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
        
        emoji = {
            'Positive': '😊',
            'Neutral': '😐',
            'Negative': '😞'
        }.get(top['label'], '')
        
        print(f"{lang_name:<10} {text[:48]:<50} "
              f"{top['label']} {emoji:<8} {top['score']*100:6.2f}%")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    import sys
    
    print("\n" + "#"*70)
    print("#  TRILINGUAL SENTIMENT CLASSIFIER - PREDICTION SCRIPT")
    print("#  Languages: English, Hindi, Telugu")
    print("#"*70)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'interactive':
            interactive_mode()
        elif command == 'examples':
            run_examples()
        else:
            # Treat as direct text prediction
            text = ' '.join(sys.argv[1:])
            result = predict_sentiment(text)
            print("\n" + json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Default: run examples
        run_examples()
