import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Initialise tokeniser and model for distil-multiBERT
model_name = "distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(
    model_name,
    output_attentions=True
).to('cpu')  # Use 'cuda' if a GPU is available

# Load human fixation datasets (L1 and L2)
human_data_l1 = pd.read_excel('/Users/rajivbains/Downloads/L1ReadingData.xlsx', sheet_name='DATA', nrows=500)
human_data_l2 = pd.read_excel('/Users/rajivbains/Downloads/L2ReadingData.xlsx', sheet_name='DATA', nrows=500)

# Load model attention datasets (English and Dutch Materials)
model_data_english = pd.read_excel('/Users/rajivbains/Downloads/EnglishMaterial.xlsx', sheet_name='ALL', nrows=500)
model_data_dutch = pd.read_excel('/Users/rajivbains/Downloads/DutchMaterials.xlsx', sheet_name='ALL', nrows=500)

# Debugging: Check column names
print("L1 Data Columns:", human_data_l1.columns)
print("English Material Columns:", model_data_english.columns)
print("L2 Data Columns:", human_data_l2.columns)
print("Dutch Material Columns:", model_data_dutch.columns)

# Ensure alignment by merging datasets on WORD_ID (since SENTENCE_ID is unavailable in human datasets)
def align_word_level_datasets(human_data, model_data):
    aligned_data = pd.merge(human_data, model_data, how='inner', on='WORD_ID')
    return aligned_data

# Align datasets at the word level
english_word_aligned = align_word_level_datasets(human_data_l1, model_data_english)
dutch_word_aligned = align_word_level_datasets(human_data_l2, model_data_dutch)

# Generate model attention scores
def compute_model_attention(dataset, text_column='WORD'):
    # Extract attention scores for each word
    attention_scores = []
    for text in dataset[text_column]:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs, output_attentions=True)
        # Use the last layer's attention and average across heads
        last_layer_attention = outputs.attentions[-1].mean(dim=1).mean().item()
        attention_scores.append(last_layer_attention)
    return attention_scores

# Validate model attention scores
print("Computing model attention scores for English dataset...")
model_data_english['MODEL_ATTENTION'] = compute_model_attention(model_data_english, text_column='WORD')
print("Computed MODEL_ATTENTION column:", model_data_english[['WORD', 'MODEL_ATTENTION']].head())

print("Computing model attention scores for Dutch dataset...")
model_data_dutch['MODEL_ATTENTION'] = compute_model_attention(model_data_dutch, text_column='WORD')
print("Computed MODEL_ATTENTION column:", model_data_dutch[['WORD', 'MODEL_ATTENTION']].head())

# Ensure alignment includes MODEL_ATTENTION
english_word_aligned = align_word_level_datasets(human_data_l1, model_data_english)
if 'MODEL_ATTENTION' not in english_word_aligned.columns:
    print("MODEL_ATTENTION column missing in english_word_aligned. Debugging required.")
    
dutch_word_aligned = align_word_level_datasets(human_data_l1, model_data_dutch)
if 'MODEL_ATTENTION' not in dutch_word_aligned.columns:
    print("MODEL_ATTENTION column missing in dutch_word_aligned. Debugging required.")

# Aggregate model data to the sentence level
def aggregate_sentence_level_data(model_data):
    sentence_level_data = model_data.groupby('SENTENCE_ID').agg({
        'IA_AREA': 'mean',
        'WORD_LENGTH': 'sum',
        'MODEL_ATTENTION': 'mean'
    }).reset_index()
    return sentence_level_data

english_sentence_level = aggregate_sentence_level_data(model_data_english)
dutch_sentence_level = aggregate_sentence_level_data(model_data_dutch)

# Function to extract attention scores
def extract_attention_scores(texts):
    attention_scores = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs, output_attentions=True)
        # Use the last layer's attention and average across heads
        last_layer_attention = outputs.attentions[-1].mean(dim=1).detach().numpy()
        attention_scores.append(last_layer_attention)
    return np.array(attention_scores)

# Compute correlations between human fixation metrics and model attention scores
def compute_correlation(data, human_metric, model_metric):
    correlation, p_value = spearmanr(data[human_metric], data[model_metric])
    return correlation, p_value

# Example correlations for word-level data
correlation_english_word, p_value_english_word = compute_correlation(english_word_aligned, 'WORD_FIXATION_COUNT', 'MODEL_ATTENTION')
correlation_dutch_word, p_value_dutch_word = compute_correlation(dutch_word_aligned, 'WORD_FIXATION_COUNT', 'MODEL_ATTENTION')

print(f"Word-Level English Correlation: {correlation_english_word}, p-value: {p_value_english_word}")
print(f"Word-Level Dutch Correlation: {correlation_dutch_word}, p-value: {p_value_dutch_word}")

# Example correlations for sentence-level data
correlation_english_sentence, p_value_english_sentence = compute_correlation(english_sentence_level, 'IA_AREA', 'MODEL_ATTENTION')
correlation_dutch_sentence, p_value_dutch_sentence = compute_correlation(dutch_sentence_level, 'IA_AREA', 'MODEL_ATTENTION')

print(f"Sentence-Level English Correlation: {correlation_english_sentence}, p-value: {p_value_english_sentence}")
print(f"Sentence-Level Dutch Correlation: {correlation_dutch_sentence}, p-value: {p_value_dutch_sentence}")

# Scatter plot of human fixation counts vs. model attention (word level)
def plot_correlation(data, human_metric, model_metric, language, level):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[human_metric], y=data[model_metric], alpha=0.6)
    plt.title(f"{level} Correlation: {human_metric} vs. {model_metric} ({language})")
    plt.xlabel(human_metric)
    plt.ylabel(model_metric)
    plt.show()

# Plot word-level correlations
plot_correlation(english_word_aligned, 'WORD_FIXATION_COUNT', 'MODEL_ATTENTION', 'English', 'Word-Level')
plot_correlation(dutch_word_aligned, 'WORD_FIXATION_COUNT', 'MODEL_ATTENTION', 'Dutch', 'Word-Level')

# Plot sentence-level correlations
plot_correlation(english_sentence_level, 'IA_AREA', 'MODEL_ATTENTION', 'English', 'Sentence-Level')
plot_correlation(dutch_sentence_level, 'IA_AREA', 'MODEL_ATTENTION', 'Dutch', 'Sentence-Level')