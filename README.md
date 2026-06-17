# Multilingual Sentiment Classifier

**Fine-tuned DistilBERT for sentiment classification across English, Hindi & Telugu**

---

## 📋 Project Overview

Fine-tuned multilingual DistilBERT model for 3-class sentiment classification (Positive, Neutral, Negative) across English, Hindi, and Telugu languages with a Flask REST API.

| Detail | Information |
|--------|-------------|
| **Model** | DistilBERT (66M parameters, multilingual-cased) |
| **Task** | 3-class Sentiment Classification |
| **Languages** | English, Hindi, Telugu |
| **Framework** | PyTorch, Hugging Face Transformers |
| **Deployment** | Flask REST API |
| **Performance** | ~91% F1 Score (weighted) |

---

## 🎯 Key Features

✅ **Trilingual Support**: English, Hindi (Devanagari), Telugu (Telugu script)  
✅ **Fine-tuned DistilBERT**: Transfer learning from multilingual pre-training  
✅ **Automatic Language Detection**: Restricts to supported languages  
✅ **REST API**: Production-ready Flask deployment  
✅ **Batch Processing**: Support for multiple predictions  
✅ **GPU Support**: Automatic CUDA detection  

---

## 🏗️ Project Structure

```
├── train.py                       # Training pipeline
├── app.py                         # Flask REST API
├── predict.py                     # Prediction script
├── requirements.txt               # Python dependencies
├── sentiment_model_final/         # Saved model + tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
├── eda_trilingual_dataset.png     # Data visualization
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Deepika1289/Multilingual-Sentiment-Analysis.git
cd Multilingual-Sentiment-Analysis

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Training

```bash
python train.py
```

### 3. Start Flask API

```bash
python app.py
```

API runs on `http://127.0.0.1:5000`

### 4. Test Predictions

```bash
# Interactive mode
python predict.py interactive

# Run examples
python predict.py examples

# Direct prediction
python predict.py "This product is amazing!"
```

---

## 🔌 API Endpoints

### POST /predict
Single text prediction
```json
{
    "text": "This product is amazing!",
    "language": "en"
}
```

**Response:**
```json
{
    "text": "This product is amazing!",
    "language": "en",
    "sentiment": "Positive",
    "confidence": 0.9823,
    "scores": {
        "Positive": 0.9823,
        "Neutral": 0.0141,
        "Negative": 0.0036
    }
}
```

### POST /predict_batch
Batch predictions for multiple texts
```json
{
    "texts": ["Excellent!", "Average.", "Terrible!"]
}
```

### GET /health
Health check endpoint

### GET /info
API information and supported languages

---

## 🧪 Usage Examples

### cURL
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "बहुत बढ़िया!", "language": "hi"}'
```

### Python
```python
import requests

response = requests.post('http://127.0.0.1:5000/predict', json={
    'text': 'ఈ ఉత్పత్తి అద్భుతమైనది!',
    'language': 'te'
})

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

---

## 📦 Dependencies

See `requirements.txt` for complete list:
- torch (2.0.1)
- transformers (4.35.2)
- flask (3.0.0)
- flask-cors (4.0.0)
- langdetect (1.0.9)
- scikit-learn, pandas, numpy, matplotlib, seaborn

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python train.py` first |
| Unsupported language error | Only English, Hindi, Telugu supported |
| CUDA out of memory | Reduce batch size or use CPU |
| Port 5000 already in use | Change port in `app.py` |

---

## 📝 License

Open-source. Feel free to use, modify, and distribute.

---

## 👩‍💼 Author

**Deepika** - Data Science  
Fine-tuning DistilBERT for multilingual NLP applications

---

**Status**: Production Ready ✅
