# Fake News Detection Using NLP

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Download the Kaggle dataset and place `Fake.csv` and `True.csv` in the `data/` folder.
5. Run the Jupyter notebook to train and save the models.
6. Launch the app: `streamlit run app.py`

## Model Comparison
| Model | Accuracy | F1-Score | AUC |
|---|---|---|---|
| SVM (TF-IDF) | 0.98 | 0.98 | 0.99 |
| LSTM (Word2Vec) | 0.96 | 0.96 | 0.98 |
| BERT (Fine-tuned) | 0.99 | 0.99 | 0.99 |
