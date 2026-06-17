# -*- coding: utf-8 -*-
"""
Trilingual Sentiment Classifier - Training Script
Languages: English, Hindi, Telugu
Model: DistilBERT (fine-tuned)
Task: 3-class Sentiment Classification (Positive, Neutral, Negative)

Author: Deepika
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "distilbert-base-multilingual-cased"
OUTPUT_DIR = "./sentiment_model_final"
LOG_DIR = "./logs"
DATA_FILE = "trilingual_sentiment.csv"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# Color palette for visualizations
PALETTE = ['#1a6b4a', '#fbbf24', '#f472b6']
LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
LANG_MAP = {'en': 'English', 'hi': 'Hindi', 'te': 'Telugu'}

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Device: {device.upper()}")
if device == 'cuda':
    print(f"   GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   Training will be fast (~2-3 minutes)")
else:
    print(f"   Training will take ~10-15 minutes")
print(f"   PyTorch: {torch.__version__}")


# ============================================================================
# STEP 1: GENERATE TRILINGUAL DATASET
# ============================================================================
def generate_dataset():
    """
    Generate balanced trilingual sentiment dataset.
    Languages: English, Hindi, Telugu
    Classes: Positive (2), Neutral (1), Negative (0)
    """
    print("\n" + "="*70)
    print("STEP 1: GENERATING TRILINGUAL DATASET")
    print("="*70)
    
    np.random.seed(42)
    
    TWEETS = {
        'en': {  # English
            2: [  # Positive
                "This product is absolutely amazing! Highly recommend 🌟",
                "Just had the best customer service experience ever!",
                "So happy with my purchase, exceeded all expectations 😊",
                "Incredible quality, will definitely buy again!",
                "Best decision I made this year, loving it!",
                "Outstanding performance, totally worth every penny",
                "Five stars all the way, absolutely love this brand",
                "Wow, just wow. This exceeded all my expectations!",
            ],
            1: [  # Neutral
                "Got my order today. It arrived on time.",
                "Product is okay, does what it's supposed to do.",
                "Delivery was standard. Nothing special to mention.",
                "It works fine. No major complaints.",
                "Average experience. Neither good nor bad.",
                "Package arrived. Will use it and see.",
            ],
            0: [  # Negative
                "Terrible quality, completely disappointed 😡",
                "Worst purchase ever! Total waste of money.",
                "Customer service was rude and unhelpful.",
                "Product broke after 2 days. Awful experience.",
                "Never buying from this brand again. Disgusting service.",
                "Complete scam! Do not buy this product!",
            ]
        },
        'hi': {  # Hindi (Devanagari)
            2: [
                "बहुत बढ��िया उत्पाद है! बहुत खुश हूँ 😊",
                "शानदार सेवा, बहुत अच्छा लगा!",
                "यह उत्पाद शानदार है, दोबारा जरूर खरीदूंगा!",
                "असाधारण गुणवत्ता, सर्वश्रेष्ठ खरीद!",
                "बहुत प्रभावशाली प्रदर्शन, पूरी तरह से संतुष्ट!",
                "यह ब्रांड वाकई बेहतरीन है!",
                "उम्मीद से भी ज्यादा अच्छा, धन्यवाद!",
            ],
            1: [
                "ठीक है, काम चलाऊ है।",
                "सामान्य अनुभव रहा।",
                "ठीक-ठाक है, कोई शिकायत नहीं।",
                "समय पर डिलीवरी हुई।",
                "औसत गुणवत्ता है।",
                "सामान्य सेवा है।",
            ],
            0: [
                "बहुत खराब है! बिल्कुल पसंद नहीं आया।",
                "पैसे की बर्बादी है।",
                "कभी मत खरीदो यह उत्पाद!",
                "बहुत निराशाजनक गुणवत्ता।",
                "सेवा बहुत खराब थी।",
                "पूरी तरह विफल, पैसे वापस करो!",
            ]
        },
        'te': {  # Telugu (Telangana/Andhra Pradesh)
            2: [
                "ఈ ఉత్పత్తి అద్భుతమైనది! నేను దీన్ని బాగా సిఫార్సు చేస్తాను 😊",
                "అద్భుత సేవ! చాలా సంతోషం!",
                "ఈ ఉత్పత్తి సుపిరితమైనది, తిరిగి కొనుక్కుంటాను!",
                "అసాధారణ నాణ్యత, చెల్లిం వెలు!",
                "చాలా చక్కగా ఉంది, సంపూర్ణంగా సంతృప్తిగా ఉన్నాను!",
                "ఈ బ్రాండ్ నిజంగా చక్కని!",
                "ఆశలకంటే చాలా బాగుంది, థాంక్యూ!",
            ],
            1: [
                "సరిగానే ఉంది, పనిచేస్తుంది.",
                "సామాన్య అనుభవం.",
                "సరిగానే ఉంది, ఎటువంటి ఫిర్యాదు లేదు.",
                "సమయానికి డెలివరీ అయింది.",
                "సామాన్య నాణ్యత.",
                "సామాన్య సేవ.",
            ],
            0: [
                "చాలా చెడ్డ ఉంది! చిట్టచివర్లకి నిరాశ చెందాను.",
                "డబ్బు వృధా!",
                "ఈ ఉత్పత్తిని ఎప్పుడూ కొనకండి!",
                "నాణ్యత చాలా చెడ్డది.",
                "సేవ చాలా చెడ్డది.",
                "సంపూర్ణ విఫలం, డబ్బు తిరిగి ఇవ్వండి!",
            ]
        }
    }
    
    rows = []
    for lang, sentiments in TWEETS.items():
        for label, tweet_list in sentiments.items():
            for tweet in tweet_list:
                # Augment dataset by repeating samples
                for _ in range(5):
                    rows.append({
                        'text': tweet,
                        'label': label,
                        'language': lang
                    })
    
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"   Total samples: {len(df):,}")
    print(f"   Languages: {', '.join([LANG_MAP[l] for l in df['language'].unique()])}")
    print(f"\n   Label distribution:")
    for label, count in sorted(df['label'].value_counts().items()):
        pct = (count / len(df)) * 100
        print(f"     {LABEL_MAP[label]:10s}: {count:4d} ({pct:5.1f}%)")
    print(f"\n   Language distribution:")
    for lang, count in df['language'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"     {LANG_MAP[lang]:10s}: {count:4d} ({pct:5.1f}%)")
    
    return df


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
def exploratory_data_analysis(df):
    """
    Perform EDA and generate visualization.
    """
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    df['sentiment'] = df['label'].map(LABEL_MAP)
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    
    # Create comprehensive EDA visualization
    fig = plt.figure(figsize=(16, 10), facecolor='#f8fafb')
    fig.suptitle('Trilingual Sentiment Dataset — EDA (English, Hindi, Telugu)',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    
    # 1. Sentiment distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df['sentiment'].value_counts()[['Positive', 'Neutral', 'Negative']]
    bars = ax1.bar(counts.index, counts.values, color=PALETTE,
                   edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha='center', fontweight='bold', fontsize=11)
    ax1.set_title('Sentiment Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_ylim(0, counts.max() * 1.2)
    
    # 2. Language distribution
    ax2 = fig.add_subplot(gs[0, 1])
    lang_counts = df['language'].value_counts()
    ax2.barh([LANG_MAP[l] for l in lang_counts.index], lang_counts.values,
             color=PALETTE[0], edgecolor='white', alpha=0.85)
    for i, val in enumerate(lang_counts.values):
        ax2.text(val + 1, i, str(val), va='center', fontsize=10)
    ax2.set_xlabel('# Samples')
    ax2.set_title('Language Distribution', fontweight='bold')
    
    # 3. Sentiment by language
    ax3 = fig.add_subplot(gs[0, 2])
    lang_sent = df.groupby(['language', 'sentiment']).size().unstack(fill_value=0)
    lang_sent.index = [LANG_MAP[l] for l in lang_sent.index]
    lang_sent[['Positive', 'Neutral', 'Negative']].plot(
        kind='bar', ax=ax3, color=PALETTE, edgecolor='white', width=0.7, rot=30)
    ax3.set_title('Sentiment by Language', fontweight='bold')
    ax3.set_ylabel('Count')
    ax3.legend(framealpha=0, fontsize=9)
    ax3.set_xlabel('')
    
    # 4. Text length distribution
    ax4 = fig.add_subplot(gs[1, 0])
    for sent, color in zip(['Positive', 'Neutral', 'Negative'], PALETTE):
        data = df[df['sentiment'] == sent]['text_length']
        ax4.hist(data, bins=15, alpha=0.65, color=color, label=sent, edgecolor='white')
    ax4.axvline(512/4, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label='DistilBERT limit (est.)')
    ax4.set_xlabel('Word Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Tweet Length Distribution', fontweight='bold')
    ax4.legend(framealpha=0, fontsize=9)
    
    # 5. Average length by sentiment
    ax5 = fig.add_subplot(gs[1, 1])
    avg_len = df.groupby('sentiment')['text_length'].mean()[['Positive', 'Neutral', 'Negative']]
    bars = ax5.bar(avg_len.index, avg_len.values, color=PALETTE,
                   edgecolor='white', width=0.5)
    for bar, val in zip(bars, avg_len.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', fontweight='bold')
    ax5.set_ylabel('Avg Word Count')
    ax5.set_title('Avg Tweet Length by Sentiment', fontweight='bold')
    
    # 6. Sample tweets
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    samples = df.groupby('sentiment').first().reset_index()[['sentiment', 'text']]
    table_data = [[row['sentiment'], row['text'][:40] + '...'] for _, row in samples.iterrows()]
    table = ax6.table(cellText=table_data, colLabels=['Sentiment', 'Sample Text'],
                      cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    ax6.set_title('Sample Tweets', fontweight='bold', pad=20)
    
    plt.savefig('eda_trilingual_dataset.png', dpi=150, bbox_inches='tight')
    print("\n✅ EDA visualization saved: eda_trilingual_dataset.png")
    
    return df


# ============================================================================
# STEP 3: SENTIMENT DATASET CLASS
# ============================================================================
class SentimentDataset(Dataset):
    """
    PyTorch Dataset wrapper for sentiment classification.
    Handles tokenization and attention masks.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of input texts
            labels: List of labels (0, 1, 2)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


# ============================================================================
# STEP 4: COMPUTE WEIGHTED CLASS LOSS
# ============================================================================
def compute_class_weights(labels):
    """
    Compute class weights for handling imbalanced datasets.
    """
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


# ============================================================================
# STEP 5: METRICS COMPUTATION
# ============================================================================
def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 score during evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
    }


# ============================================================================
# STEP 6: TRAINING PIPELINE
# ============================================================================
def train_model(df):
    """
    Fine-tune DistilBERT on trilingual sentiment data.
    """
    print("\n" + "="*70)
    print("STEP 3: TOKENIZATION & DATASET PREPARATION")
    print("="*70)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ Tokenizer loaded! Vocabulary size: {tokenizer.vocab_size:,} tokens")
    
    # Tokenization demo
    print("\n" + "-"*70)
    print("  TOKENIZATION EXAMPLES")
    print("-"*70)
    examples = [
        "I love this product! Amazing quality 😊",
        "बहुत बढ़िया उत्पाद है!",
        "ఈ ఉత్పత్తి అద్భుతమైనది!",
    ]
    for text in examples:
        tokens = tokenizer.tokenize(text)
        print(f"Text   : {text}")
        print(f"Tokens : {tokens}")
        print(f"Length : {len(tokens)} tokens\n")
    
    # Train/Val/Test split
    print("\n" + "="*70)
    print("STEP 4: TRAIN/VAL/TEST SPLIT")
    print("="*70)
    
    X, y = df['text'].values, df['label'].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    print(f"\n✅ Datasets created:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Val  : {len(val_dataset):,} samples")
    print(f"   Test : {len(test_dataset):,} samples")
    
    # Compute class weights for balanced loss
    class_weights = compute_class_weights(y_train)
    print(f"\n✅ Class weights (for weighted loss):")
    for i, weight in enumerate(class_weights):
        print(f"   {LABEL_MAP[i]}: {weight:.4f}")
    
    # Load model
    print("\n" + "="*70)
    print("STEP 5: LOAD DISTILBERT MODEL")
    print("="*70)
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        label2id={'Negative': 0, 'Neutral': 1, 'Positive': 2}
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Model loaded!")
    print(f"   Total parameters    : {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device              : {next(model.parameters()).device}")
    
    # Training arguments
    print("\n" + "="*70)
    print("STEP 6: FINE-TUNING (3 EPOCHS)")
    print("="*70)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_dir=LOG_DIR,
        logging_steps=20,
        report_to='none',
        fp16=torch.cuda.is_available(),
    )
    
    # Trainer with custom loss weights
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"\n✅ Trainer initialized!")
    print(f"   Epochs        : {NUM_EPOCHS}")
    print(f"   Batch size    : {BATCH_SIZE}")
    print(f"   Learning rate : {LEARNING_RATE}")
    print(f"   Warmup steps  : {WARMUP_STEPS}")
    print(f"   GPU enabled   : {torch.cuda.is_available()}")
    
    print("\n🚀 Starting fine-tuning...\n")
    train_result = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Total time: {train_result.metrics['train_runtime']:.0f} seconds")
    print(f"   Speed: {train_result.metrics['train_samples_per_second']:.1f} samples/sec")
    
    # Save model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   Model saved to: {OUTPUT_DIR}")
    
    return trainer, test_dataset, y_test, tokenizer


# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
def evaluate_model(trainer, test_dataset, y_test):
    """
    Evaluate model on test set and generate evaluation plots.
    """
    print("\n" + "="*70)
    print("STEP 7: EVALUATION ON TEST SET")
    print("="*70)
    
    predictions_output = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions_output.predictions, axis=1)
    y_true = y_test
    
    print(f"\n✅ Evaluation Results:")
    print(f"   Accuracy : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"   F1 Score : {f1_score(y_true, y_pred, average='weighted')*100:.2f}%")
    print(f"\n   Per-class metrics:")
    print(classification_report(y_true, y_pred,
                                target_names=['Negative', 'Neutral', 'Positive'],
                                digits=3))
    
    # Evaluation visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#f8fafb')
    fig.suptitle('Trilingual Sentiment Classifier — Test Set Evaluation',
                 fontweight='bold', fontsize=14)
    
    # Confusion matrix
    ax1 = axes[0]
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                linewidths=0.5, linecolor='white', cbar_kws={'label': '%'},
                ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.set_title('Confusion Matrix (%)', fontweight='bold')
    
    # Per-class F1
    ax2 = axes[1]
    per_class_f1 = [f1_score(y_true == i, y_pred == i) for i in range(3)]
    bars = ax2.bar(['Negative', 'Neutral', 'Positive'], per_class_f1,
                   color=PALETTE, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, per_class_f1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Per-Class F1 Score', fontweight='bold')
    ax2.axhline(0.9, color='gray', linestyle='--', linewidth=1, alpha=0.6,
                label='0.90 target')
    ax2.legend(framealpha=0)
    
    plt.tight_layout()
    plt.savefig('evaluation_test_set.png', dpi=150, bbox_inches='tight')
    print("\n✅ Evaluation plots saved: evaluation_test_set.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "#"*70)
    print("#  TRILINGUAL SENTIMENT CLASSIFIER - TRAINING PIPELINE")
    print("#  Languages: English, Hindi, Telugu")
    print("#  Model: DistilBERT (fine-tuned)")
    print("#  Task: 3-class Sentiment (Positive, Neutral, Negative)")
    print("#"*70)
    
    # Step 1: Generate dataset
    df = generate_dataset()
    
    # Step 2: EDA
    df = exploratory_data_analysis(df)
    
    # Step 3-6: Train model
    trainer, test_dataset, y_test, tokenizer = train_model(df)
    
    # Step 7: Evaluate
    evaluate_model(trainer, test_dataset, y_test)
    
    print("\n" + "#"*70)
    print("#  ✅ TRAINING COMPLETE!")
    print("#"*70)
    print(f"\nModel saved at: {OUTPUT_DIR}")
    print(f"Logs saved at: {LOG_DIR}")
    print("\nNext steps:")
    print(f"  1. Run Flask API: python app.py")
    print(f"  2. Test predictions: python predict.py")
