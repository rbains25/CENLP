Multilingual BERT and Human Attention Analysis

This project investigates the alignment between human fixation patterns and model attention scores using DistilBERT for English and Dutch texts. By leveraging the GECO eye-tracking dataset, the study analyses correlations between human fixation metrics and attention mechanisms at word and sentence levels for both native (L1) and non-native (L2) readers.

Table of Contents
1. Project Overview
2.	Dependencies
3.	Setup Instructions
4.	Data Access
5.	Running the Scripts
6.	Adjusting Path Names
7.	Results

Project Overview

The code evaluates correlations between human fixation metrics (e.g. WORD_FIXATION_COUNT) and attention scores derived from the DistilBERT model. Attention scores are aggregated for word- and sentence-level comparisons. This project uses multilingual datasets (English and Dutch) from the GECO eye-tracking dataset.

Dependencies

Install the required libraries by running:

pip install -r requirements.txt

requirements.txt:

pandas
numpy
torch
transformers
scipy
matplotlib
seaborn
openpyxl

These dependencies include:
- pandas and openpyxl for data manipulation and reading Excel files.
- torch and transformers for utilising DistilBERT.
- matplotlib and seaborn for visualisation.
- scipy for correlation analysis.

Setup Instructions
1.	Download the method.py file from the github repository: https://github.com/rbains25/CENLP.git 

2.	Install the required dependencies as outlined above.
3.	Download the GECO eye-tracking datasets:
- URL: [GECO dataset](https://expsy.ugent.be/downloads/geco/).
- Place the Excel files in a directory of your choice.
4.	Modify the DutchMaterials.xlsx file:
- Open the ALL sheet.
- Rename the IA_ID column to WORD_ID.

Data Access

The data used in this study can be downloaded from the official GECO dataset website:
GECO Dataset [Download](https://expsy.ugent.be/downloads/geco/)

Place the downloaded Excel files (L1ReadingData.xlsx, L2ReadingData.xlsx, EnglishMaterial.xlsx, DutchMaterials.xlsx) in your prefered directory.

Running the Scripts
Preprocessing and Alignment: to align datasets and compute word-level or sentence-level correlations, run: python method.py

Key Outputs:
1.	Aligned datasets: Word-level datasets aligned by WORD_ID.
2.	Attention scores: Computes and adds the MODEL_ATTENTION column for English and Dutch materials.
3.	Correlation results:
- Word-level: Correlations between WORD_FIXATION_COUNT and MODEL_ATTENTION.
- Sentence-level: Correlations between IA_AREA and MODEL_ATTENTION.

Visualisations: The script generates scatter plots visualizing correlations between human fixation metrics and model attention scores.

Adjusting Path Names

To ensure the scripts work on your system:
1.	Locate the following lines in method.py:

human_data_l1 = pd.read_excel('/Users/rajivbains/Downloads/L1ReadingData.xlsx', sheet_name='DATA', nrows=500)
human_data_l2 = pd.read_excel('/Users/rajivbains/Downloads/L2ReadingData.xlsx', sheet_name='DATA', nrows=500)
model_data_english = pd.read_excel('/Users/rajivbains/Downloads/EnglishMaterial.xlsx', sheet_name='ALL', nrows=500)
model_data_dutch = pd.read_excel('/Users/rajivbains/Downloads/DutchMaterials.xlsx', sheet_name='ALL', nrows=500)

2.	Replace /Users/rajivbains/Downloads/ with the directory where the datasets are stored. For example:

human_data_l1 = pd.read_excel('./data/L1ReadingData.xlsx', sheet_name='DATA', nrows=500)

3.	Ensure all file paths are adjusted before running the script.

Results

The script produces:
1.	Correlation Results:
- Spearman’s rank correlation coefficients and p-values for both English and Dutch datasets.
2.	Plots:
- Scatter plots showing relationships between human fixation metrics and model attention scores.
3.	Aligned Datasets:
- Saved aligned datasets.

Contact:
For questions or issues, contact Rajiv Bains at rajiv.bains@uzh.ch.
