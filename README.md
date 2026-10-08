# Product Reviews NLP

Exploratory analysis and modeling of shopper reviews: rating patterns, text cleaning, top-rated product mining, bag-of-words features, a supervised model that predicts ratings, and unsupervised learning on review text.

**Dataset:** [Grammar and Online Product Reviews](https://www.kaggle.com/datafiniti/grammar-and-online-product-reviews) (Datafiniti, Kaggle)

## Contents

| File | Description |
|---|---|
| `Analysing_Product_Reviews_by_Shoppers.ipynb` | End-to-end notebook: EDA → cleaning → features → models → insights |

## Approach

1. **EDA:** rating distribution and review length
2. **Text cleaning:** NLTK stop-word removal and normalization
3. **Top-rated products:** identify leading products and mine reasons from top-rated comments
4. **Features:** bag-of-words representation
5. **Supervised model:** predict ratings on a train/test split, scored with accuracy
6. **Unsupervised learning:** KMeans clustering and LDA topic modeling (gensim) on review text

## Run

Open the notebook in Jupyter or Google Colab. It uses the standard Python data stack:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn nltk gensim wordcloud
```

Download the dataset from Kaggle and point the notebook's data path at the CSV.
