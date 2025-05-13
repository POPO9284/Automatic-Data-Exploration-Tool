# main.py
from flask import Flask, jsonify, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from uuid import uuid4  
from wordcloud import WordCloud
from collections import defaultdict
from openai import OpenAI
from nltk.util import ngrams
from scipy.stats import kstest, uniform
from gensim import corpora
from gensim.models import LdaModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from itertools import combinations
import os
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import logging
import scipy.stats as stats
import itertools
import re
import spacy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['IMAGE_FOLDER'] = 'static/images'

os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Delete old dataset on startup
pkl_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_dataframe.pkl')
if os.path.exists(pkl_path):
    os.remove(pkl_path)

# Dataset Statistic
def get_dataset_statistics(dataframe):
    num_rows, num_cols = dataframe.shape
    missing_cells = dataframe.isnull().sum().sum()
    duplicate_rows = dataframe.duplicated().sum()
    memory_size = dataframe.memory_usage(deep=True).sum()

    statistics = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "memory_size": round(memory_size / (1024 * 1024), 4)
    }

    return statistics

# Missing Values Distribution
def get_missing_value_distribution(dataframe):
    missing_values = []

    for feature in dataframe.columns:
        count = dataframe[feature].isnull().sum()

        if count > 1:
            missing_values.append({
                "feature": feature,
                "missing_value": count
            })
        else:
            continue
            
    return missing_values

# Missing Values Bar Chart 
def generate_missing_values_barchart(missing_data):
    if not missing_data:
        return None
    
    plot_paths = []
    
    dataframe_missing = pd.DataFrame(missing_data)
    
    plt.figure(figsize=(4, 3))
    sns.barplot(x='missing_value', y='feature', data=dataframe_missing, palette='CMRmap',width=0.5)
    plt.xlabel("")
    plt.ylabel("")
    plt.yticks(fontsize=7)

    plt.tight_layout()

    plot_paths.append(save_plot("missing_bar.png"))

    return plot_paths

# Drop Meaningless Feature
def drop_meaningless_features(dataframe):
    feature_to_drop = []
    
    for feature in dataframe.columns:
        # Drop unnamed feature or all NaN values
        if "Unnamed" in feature or dataframe[feature].isnull().all():
            feature_to_drop.append(feature)
            continue
        
        # Drop sequential integer feature
        if pd.api.types.is_integer_dtype(dataframe[feature]):
            if dataframe[feature].is_monotonic_increasing or dataframe[feature].is_monotonic_decreasing:
                if dataframe[feature].nunique() == len(dataframe) and dataframe[feature].diff().nunique() == 1:
                    feature_to_drop.append(feature)
                    continue

        # Drop alphanumeric feature
        if pd.api.types.is_string_dtype(dataframe[feature]) or dataframe[feature].dtype == 'object':
            non_missing_values = dataframe[feature].dropna()
            
            contains_alpha = non_missing_values.str.contains(r'[A-Za-z]', na=False).any()
            contains_numeric = non_missing_values.str.contains(r'[0-9]', na=False).any()
            contains_whitespace = non_missing_values.str.contains(r'\s', na=False).any()
            
            if contains_alpha and contains_numeric and not contains_whitespace:
                feature_to_drop.append(feature)

    
    cleaned_dataframe = dataframe.drop(columns=feature_to_drop)
    return cleaned_dataframe, feature_to_drop

# Feature Grouping
def group_dataframe_features(dataframe):
    grouped_features = {
        'numeric': [],
        'text': [],
        'categorical': [],
        'date': [],
        'other': []
    }

    categorical_threshold = 0.05  

    for feature in dataframe.columns:
        series = dataframe[feature]

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            grouped_features['numeric'].append(feature)

        # Datetime (direct detection or parsed)
        elif pd.api.types.is_datetime64_any_dtype(series):
            grouped_features['date'].append(feature)
        
        # Attempt to parse as datetime for string/object features
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            try:
                # Use flexible date parsing with infer_datetime_format and dayfirst
                parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True, dayfirst=True)

                # Check if parsed successfully (at least 80% valid dates)
                if parsed.notna().mean() > 0.8:
                    grouped_features['date'].append(feature)
                else:
                    # If not valid, check for categorical or text
                    unique_ratio = series.nunique() / len(series)
                    if unique_ratio < categorical_threshold:
                        grouped_features['categorical'].append(feature)
                    else:
                        grouped_features['text'].append(feature)
            except Exception:
                grouped_features['text'].append(feature)
        else:
            grouped_features['other'].append(feature)

    return grouped_features

# Format decimal points
def fmt(x):
    if pd.notnull(x):
        return f"{x:.4f}" 
    else: 
        None
        
# Numerical Statistics
def calculate_numeric_statistics(dataframe, numeric_features):
    statistics = []

    for feature in numeric_features:
        series = dataframe[feature]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()

        statistics.append({
            "var_name": feature,
            "minimum": fmt(series.min()),
            "maximum": fmt(series.max()),
            "Q1": fmt(q1),
            "Q3": fmt(q3),
            "mean": fmt(series.mean()),
            "median": fmt(series.median()),
            "standard_dev": fmt(series.std()),
            "variance": fmt(series.var()),
            "outlier_count": outlier_count
        })
        
    return statistics

# Pearson Correlation Matrix
def calculate_pearson_correlation(dataframe, numeric_features):
    if len(numeric_features) < 2:
        return None
    
    clean_dataframe = dataframe[numeric_features].dropna()
    correlation_matrix = clean_dataframe.corr(method='pearson')
    correlation_dict = correlation_matrix.round(4).to_dict(orient="index")

    return correlation_dict

# Categorical Statistics
def calculate_categoric_statistics(dataframe, categoric_features):
    statistics = {}

    for feature in categoric_features:
        frequency = dataframe[feature].value_counts()
        percentage = dataframe[feature].value_counts(normalize=True)
        mode = dataframe[feature].mode().iloc[0]

        statistics[feature] = {
            "mode": mode,
            "values": []
        }
        
        for value, count in frequency.items():
            statistics[feature]["values"].append({
                "value": value,
                "frequency": count,
                "percentage": fmt(percentage[value])
            })
    
    return statistics

# Chi-Square Test
def calculate_chi_square_p_values(dataframe, categoric_features):
    if len(categoric_features) < 2:
        return None

    chi_matrix = pd.DataFrame(index=categoric_features, columns=categoric_features)

    for var1, var2 in combinations(categoric_features, 2):
        contingency_table = pd.crosstab(dataframe[var1], dataframe[var2])
        _, p_value, _, _ = stats.chi2_contingency(contingency_table)

        chi_matrix.loc[var1, var2] = round(p_value, 4)
        chi_matrix.loc[var2, var1] = round(p_value, 4)

    chi_results = chi_matrix.to_dict(orient="index")
    
    return chi_results  

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
spacy_stopwords = nlp.Defaults.stop_words

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Text Preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Remove stopwords and lemmatize words
def remove_stopwords_lemmatize(text):
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.text not in spacy_stopwords]

    return " ".join(words)

# Tokenize Text
def tokenize_text(text, tokenize_type="word"):
    if tokenize_type == "sentence":
        return sent_tokenize(text)
    elif tokenize_type == "word":
        return word_tokenize(text)
    else:
        raise ValueError("tokenize_type must be 'sentence' or 'word'")

# Sentiment Analysis using VADER    
def analyze_sentiment_vader(text):
    scores = vader_analyzer.polarity_scores(text)
    
    if scores["compound"] > 0:
        return "Positive"
    elif scores["compound"] < 0:
        return "Negative"
    else:
        return "Neutral"
    
# Text Analysis
def calculate_textural_statistics(dataframe, text_features, processed_text):
    statistics = {}

    for feature in text_features:
        if feature not in dataframe.columns:
            continue  

        sentence_length = dataframe[feature].dropna().apply(lambda x: len(x.split()))
        min_length = sentence_length.min()
        mean_length = int(sentence_length.mean())
        max_length = sentence_length.max()
        unique_sentences = dataframe[feature].nunique()

        # Sentiment classification
        sentiments = processed_text[feature].apply(analyze_sentiment_vader)
        sentiment_counts = sentiments.value_counts().to_dict()

        statistics[feature] = {
            "min_sentence_length": min_length,
            "average_sentence_length": mean_length,
            "max_sentence_length": max_length,
            "unique_sentence_count": unique_sentences,
            "positive_sentiment_count": sentiment_counts.get("Positive", 0),
            "negative_sentiment_count": sentiment_counts.get("Negative", 0),
            "neutral_sentiment_count": sentiment_counts.get("Neutral", 0)
        }

    return statistics

# Generate n-grams
def get_ngrams(text, n):
    tokens = word_tokenize(text)
    return list(ngrams(tokens, n))

# Word Frequency Analysis
def perform_word_ngram_analysis(cleaned_text, tokenized_word, text_features):
    top_words = {}
    top_two_words = {}
    top_three_words = {}

    for feature in text_features:
        if feature not in cleaned_text or feature not in tokenized_word:
            continue

        # Word Frequency Analysis
        all_words = [word for sublist in tokenized_word[feature] for word in sublist]
        word_freq = Counter(all_words)
        top_10_words = word_freq.most_common(10)

        # 2-Gram Frequency Analysis
        all_2grams = [gram for text in cleaned_text[feature] for gram in get_ngrams(text, 2)]
        two_gram_freq = Counter(all_2grams)
        top_10_2grams = two_gram_freq.most_common(10)

        # 3-Gram Frequency Analysis
        all_3grams = [gram for text in cleaned_text[feature] for gram in get_ngrams(text, 3)]
        three_gram_freq = Counter(all_3grams)
        top_10_3grams = three_gram_freq.most_common(10)

        top_words[feature] = [{"word": word, "count": count} for word, count in top_10_words]
        top_two_words[feature] = [{"bigram": " ".join(gram), "count": count} for gram, count in top_10_2grams]
        top_three_words[feature] = [{"trigram": " ".join(gram), "count": count} for gram, count in top_10_3grams]

    return top_words, top_two_words, top_three_words

# Topic Analysis
def perform_topic_modeling(tokenized_word, text_features):
    topic_results = {}

    for feature in text_features:
        if feature not in tokenized_word or tokenized_word[feature].dropna().empty:
            continue

        processed_sentence = tokenized_word[feature].dropna().tolist()
        dictionary = corpora.Dictionary(processed_sentence)
        corpus = [dictionary.doc2bow(text) for text in processed_sentence]

        lda_model = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=10)
        topics = lda_model.show_topics(num_topics=1, num_words=3, formatted=False)

        structured_topics = []
        for topic_id, probability in topics:
            topic_data = {
                "topic_id": topic_id,
                "words": [{"word": word, "probability": round(prob, 4)} for word, prob in probability]
            }
            structured_topics.append(topic_data)

        topic_results[feature] = structured_topics

    return topic_results

# Alert Box
def alert_box(dataframe, numeric_features, categoric_features, correlation):
    total_rows = len(dataframe)
    alert = {}

    # Missing Values
    missing_values = dataframe.isnull().sum()
    missing_features = missing_values[missing_values > 0]
    if not missing_features.empty:
        missing_percentage = (missing_features / total_rows) * 100
        alert["Missing Features"] = {
            feature: {"count": int(missing_features[feature]), "percentage": round(missing_percentage[feature], 2)}
            for feature in missing_features.index
        }

    # Zero Values 
    numeric_dataframe = dataframe[numeric_features]
    zero_values = (numeric_dataframe == 0).sum()
    zero_features = zero_values[zero_values > total_rows * 0.2]  
    if not zero_features.empty:
        zero_percentage = (zero_features / total_rows) * 100
        alert["Zero Features"] = {
            feature: {"count": int(zero_features[feature]), "percentage": round(zero_percentage[feature], 2)}
            for feature in zero_features.index
        }

    # Unique Values 
    unique_counts = dataframe.nunique()
    unique_features = unique_counts[unique_counts == 1].index.tolist()
    if unique_features:
        alert["Unique Features"] = unique_features

    # Duplicate Observations
    duplicate_rows = dataframe.duplicated().sum()
    if duplicate_rows > 0:
        duplicate_percentage = (duplicate_rows / total_rows) * 100
        alert["Duplicate Rows"] = {"count": int(duplicate_rows), "percentage": round(duplicate_percentage, 2)}

    # Imbalanced 
    imbalanced_features = {}
    for feature in categoric_features:
        value_counts = dataframe[feature].value_counts(normalize=True)
        if value_counts.iloc[0] > 0.7: 
            count = dataframe[feature].value_counts().iloc[0] 
            percentage = value_counts.iloc[0] * 100
            imbalanced_features[feature] = {"count": int(count), "percentage": round(percentage, 2)}

    if imbalanced_features:
        alert["Imbalanced Features"] = imbalanced_features

    # Skewed Features (|skewness| > 1)
    skewness = numeric_dataframe.skew().round(4)
    alert["Skewness"] = skewness.to_dict()
    highly_skewed_features = skewness[abs(skewness) > 1].index.tolist()

    if highly_skewed_features:
        alert["Highly Skewed Features"] = highly_skewed_features

    # Uniform Features (p-value > 0.05)
    uniform_features = [] 
    for feature in numeric_dataframe.columns:
        d_stat, p_value = kstest(numeric_dataframe[feature], uniform.cdf, args=(numeric_dataframe[feature].min(), numeric_dataframe[feature].max() - numeric_dataframe[feature].min()))
        
        if p_value > 0.05:  # Fail to reject uniformity
            uniform_features.append(feature)

    if uniform_features:
        alert["Uniform Features"] = uniform_features

    # Correlated Features (Correlation > 0.8)
    if correlation is not None:
        correlated_features = {}
        for feature1, correlations in correlation.items():
            for feature2, corr_value in correlations.items():
                if feature1 != feature2 and abs(corr_value) > 0.8:
                    correlated_features[f"{feature1} & {feature2}"] = round(corr_value, 2)

        if correlated_features:
            alert["Highly Correlated Features"] = correlated_features

    return alert

# Save Plot
def save_plot(filename):
    plot_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
    plt.savefig(plot_path)
    plt.clf()
    return filename

# Visualization Numerical Histogram
def generate_numeric_histrogram(dataframe, numeric_features):
    plot_paths = []
    
    for feature in (numeric_features):
        plt.figure(figsize=(2, 2))
        sns.histplot(dataframe[feature].dropna(), kde=True, color='blue')
        plt.xlabel(feature, fontsize=7)
        plt.ylabel("")

        plt.tight_layout()
        
        plot_paths.append(save_plot(f'{feature}_hist.png'))
    
    return plot_paths

# Visualization Numerical Boxplot
def generate_numeric_boxplot(dataframe, numeric_features):
    plot_paths = []
    
    for feature in (numeric_features):
        plt.figure(figsize=(2, 2))
        sns.boxplot(dataframe[feature], color='orange')

        plt.title(feature, fontsize=7)
        plt.ylabel("")
        plt.tight_layout()

        plot_paths.append(save_plot(f"{feature}_box.png"))
    
    return plot_paths

# Visualization Numerical Pairplot
def generate_numeric_pairplot(dataframe, numeric_features):
    plot_paths = []

    if not numeric_features:
        return None

    sns.pairplot(dataframe[numeric_features], 
                 plot_kws={'alpha': 0.7, 'edgecolor': None, 'color': '#5799c6'},
                 height=2)  # adjust size per plot if needed
    plt.tight_layout()

    plot_paths.append(save_plot(f"pairplot.png"))   

    return plot_paths

# Visualization Numerical Heatmap
def generate_numeric_heatmap(dataframe, numeric_features):
    num_vars_count = len(numeric_features)
    plot_paths = []

    if num_vars_count > 1:
        # Compute the correlation matrix
        corr = dataframe[numeric_features].corr()

        # Replace Inf and NaN values with 0
        corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Drop constant columns and rows with all zeros
        non_zero_cols = (corr != 0).any(axis=0)
        non_zero_rows = (corr != 0).any(axis=1)

        corr = corr.loc[non_zero_rows, non_zero_cols]

        # Check if the correlation matrix still has enough data to plot
        if corr.shape[0] > 1 and corr.shape[1] > 1:
            cluster_grid = sns.clustermap(
                corr, 
                method='average',  
                metric='correlation', 
                cmap='Blues', 
                vmin=-1, vmax=1, 
                annot=True, fmt=".2f", 
                figsize=(8, 5),
                cbar_pos=None,
            )

            plt.draw()
            cluster_grid.ax_heatmap.set_xticklabels(
                cluster_grid.ax_heatmap.get_xticklabels(), 
                rotation=25, ha='right', fontsize=8
            )
            cluster_grid.ax_heatmap.set_yticklabels(
                cluster_grid.ax_heatmap.get_yticklabels(), 
                rotation=25, va='center', fontsize=8
            )
            plt.tight_layout()

            plot_paths.append(save_plot("heatmap.png"))
    
    return plot_paths

# Visualization Categorical Bar Chart
def generate_categoric_barchart(dataframe, categoric_features):
    plot_paths = []
    
    for feature in categoric_features:
        plt.figure(figsize=(3, 2))
        unique_categories = dataframe[feature].dropna().unique()
        palette = sns.color_palette('CMRmap', len(unique_categories))
        
        sns.countplot(y=feature, data=dataframe, hue=feature, palette=palette, order=unique_categories, dodge=False, legend=False,width=0.5)
        plt.title(feature, fontsize=7)
        plt.xlabel('')
        plt.ylabel('')

        plt.yticks(fontsize=7)
        plt.tight_layout()

        plot_paths.append(save_plot(f"{feature}_bar.png"))

    return plot_paths

# Categorical Pie Chart
def generate_categoric_piechart(dataframe, categoric_features):
    plot_paths = []

    for feature in categoric_features:
        value_counts = dataframe[feature].value_counts()
        top_4 = value_counts.nlargest(4)
        other_sum = value_counts.iloc[4:].sum()

        if other_sum > 0:
            top_4['Other'] = other_sum

        labels = top_4.index
        sizes = top_4.values
        pie_palette = sns.color_palette('coolwarm', len(top_4))

        plt.figure(figsize=(3, 3))
        wedges, texts, autotexts = plt.pie(
            sizes,
            colors=pie_palette,
            startangle=90,
            labels=None,  # Don't label with names here
            autopct='%1.1f%%',  # Show percentage on the pie slices
            textprops={'fontsize': 7}
        )

        # Legend with category name only (no values)
        plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)

        plt.title(f'{feature}', fontsize=9)
        plt.ylabel('')

        plt.tight_layout()

        plot_paths.append(save_plot(f"{feature}_pie.png"))

    return plot_paths

# WordCloud
def generate_text_wordcloud(tokenized_word, text_features):
    plot_paths = []

    for feature in text_features:
        words = [word for sublist in tokenized_word[feature] for word in sublist]
        text_data = " ".join(words)

        # Check if text_data is empty before generating word cloud
        if not text_data.strip():
            continue

        wordcloud = WordCloud(
            width=600, 
            height=300, 
            background_color='white',
            stopwords=set()
        ).generate(text_data)

        plt.figure(figsize=(3, 2))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{feature}', fontsize=7, pad=10)
        plt.tight_layout()

        plot_paths.append(save_plot(f"{feature}_wordcloud.png"))
    
    return plot_paths

# Unigram Bar Chart
def generate_unigram_barchart(unigram, text_features):
    plot_paths = []
    
    for feature in text_features:
        words = [item["word"] for item in unigram[feature]]
        counts = [item["count"] for item in unigram[feature]]

        # Plot the bar chart
        plt.figure(figsize=(4,2))
        plt.barh(words, counts, color='purple')
        plt.xlabel('Frequency', fontsize=7)
        plt.title(f'Top 10 Most Common Words in {feature}', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=7)

        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_paths.append(save_plot(f"{feature}_bar.png"))

    return plot_paths

# Bigram Bar Chart
def generate_bigram_barchart(bigrams, text_features):
    plot_paths = []
    
    for feature in text_features:
        bigram = [item["bigram"] for item in bigrams[feature]]
        counts = [item["count"] for item in bigrams[feature]]

        # Plot the bar chart
        plt.figure(figsize=(4, 2))
        plt.barh(bigram, counts, color='purple')
        plt.xlabel('Frequency', fontsize=7)
        plt.title(f'Top 10 Most Common Two-word in {feature}', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=7)
        
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_paths.append(save_plot(f"{feature}_bigram.png"))

    return plot_paths

# Trigram Bar Chart
def generate_trigram_barchart(trigrams, text_features):
    plot_paths = []
    
    for feature in text_features:
        trigram = [item["trigram"] for item in trigrams[feature]]
        counts = [item["count"] for item in trigrams[feature]]

        # Plot the bar chart
        plt.figure(figsize=(4,2))
        plt.barh(trigram, counts, color='purple')
        plt.xlabel('Frequency', fontsize=7)
        plt.title(f'Top 10 Most Common Three-word in {feature}', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=7)
        
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_paths.append(save_plot(f"{feature}_trigram.png"))

    return plot_paths

# Data Summary    
def generate_summary(dataset_statistics, numeric_statistics, correlation, chi_square, alert, text_statistics, unigram, bigram, trigram):
    summary = {}

    # Dataset info
    summary["dataset_overview"] = {
        "num_rows": dataset_statistics["num_rows"],
        "num_cols": dataset_statistics["num_cols"]
    }

    # Missing values
    if dataset_statistics["missing_cells"] > 0:
        summary["missing_values"] = dataset_statistics["missing_cells"]

    # Extreme values
    summary["extreme_features"] = [stat["var_name"] for stat in numeric_statistics if stat["outlier_count"] > 0]

    # Skewed features
    if "Highly Skewed Features" in alert:
        summary["skewed_features"] = alert["Highly Skewed Features"]

    # Uniform features
    if "Uniform Features" in alert:
        summary["uniform_features"] = alert["Uniform Features"]

    # Correlation
    correlation_info = []
    if correlation:
        seen_pairs = set()

        for f1, inner in correlation.items():
            for f2, corr in inner.items():
                if f1 != f2 and abs(corr) > 0.7:
                    pair = tuple(sorted((f1, f2)))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        direction = "positive" if corr > 0 else "negative"
                        correlation_info.append({
                            "feature_1": pair[0],
                            "feature_2": pair[1],
                            "direction": direction
                        })

    if correlation_info:
        summary["correlation"] = correlation_info

    # Imbalanced features
    if "Imbalanced Features" in alert:
        summary["imbalanced_features"] = list(alert["Imbalanced Features"].keys())

    # Significant association (Chi-Square)
    if chi_square:
        significant_association = []
        seen_pairs = set()

        for var1, others in chi_square.items():
            for var2, p_val in others.items():
                if var1 != var2 and p_val <= 0.05:
                    # Sort pair to avoid duplicates
                    pair = tuple(sorted((var1, var2)))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        significant_association.append(f"{pair[0]} & {pair[1]}")

        if significant_association:
            summary["significant_association"] = significant_association
            
    # Sentiment distribution
    sentiment_info = {}
    for feature, stats in text_statistics.items():
        sentiment_info[feature] = {
            "positive": stats["positive_sentiment_count"],
            "negative": stats["negative_sentiment_count"],
            "neutral": stats["neutral_sentiment_count"]
        }
    if sentiment_info:
        summary["sentiment_distribution"] = sentiment_info

    # Unigram/Bigram/Trigram
    if unigram:
        summary["unigram"] = {
            feature: [entry["word"] for entry in entries[:3]] for feature, entries in unigram.items()
        }
    if bigram:
        summary["bigram"] = {
            feature: [entry["bigram"] for entry in entries[:3]] for feature, entries in bigram.items()
        }
    if trigram:
        summary["trigram"] = {
            feature: [entry["trigram"] for entry in entries[:3]] for feature, entries in trigram.items()
        }

    return summary


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    # Check if user clicked "Upload Dataset"
    if request.args.get('reset') == '1':
        pkl_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_dataframe.pkl')
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
        return render_template('index.html')

    # POST request - save uploaded file
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = secure_filename(file.filename)

        if not filename.endswith('.csv'):
            return jsonify({"error": "Invalid file type"}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            dataframe = pd.read_csv(file_path)
            dataframe.to_pickle('static/files/uploaded_dataframe.pkl')
            return jsonify({"redirect": url_for('statistics')})  
        except Exception as e:
            return jsonify({"error": f"Error processing file: {e}"}), 500

    # GET request - check if preview data exists
    preview_html = None
    if os.path.exists('static/files/uploaded_dataframe.pkl'):
        df = pd.read_pickle('static/files/uploaded_dataframe.pkl')
        preview_html = df.head(10).to_html(classes='preview-table-custom', index=False)
    
    return render_template('index.html', preview_html=preview_html)

@app.route('/cancel_upload', methods=['POST'])
def cancel_upload():
    try:
        os.remove('static/files/uploaded_dataframe.pkl')
    except FileNotFoundError:
        pass 
    return jsonify({"message": "Upload canceled"})

@app.route('/check_dataset')
def check_dataset():
    exists = os.path.exists('static/files/uploaded_dataframe.pkl')
    return jsonify({"exists": exists})

@app.route('/statistics', methods=['GET'])
def statistics():
    try:
        df = pd.read_pickle('static/files/uploaded_dataframe.pkl')
        dataset_statistics = get_dataset_statistics(df)
        missing_count = get_missing_value_distribution(df)
        dataframe_cleaned, features_dropped = drop_meaningless_features(df)
        grouped_features = group_dataframe_features(dataframe_cleaned)

        numeric_features = grouped_features['numeric']
        categoric_features = grouped_features['categorical']
        text_features = grouped_features['text']

        numeric_statistics = calculate_numeric_statistics(df, numeric_features)
        pearson_correlation = calculate_pearson_correlation(df, numeric_features)
        categoric_statistics = calculate_categoric_statistics(df, categoric_features)
        chi_square = calculate_chi_square_p_values(df, categoric_features)

        pre_text = {}
        cleaned_text = {}
        tokenized_sent = {}
        tokenized_word = {}

        for feature in text_features:
            pre_text[feature] = df[feature].dropna().apply(preprocess_text)
            cleaned_text[feature] = pre_text[feature].apply(remove_stopwords_lemmatize)
            tokenized_sent[feature] = cleaned_text[feature].apply(lambda x: tokenize_text(x, "sentence"))
            tokenized_word[feature] = cleaned_text[feature].apply(lambda x: tokenize_text(x, "word"))

        text_statistics = calculate_textural_statistics(df, text_features, pre_text)
        unigram, bigram, trigram = perform_word_ngram_analysis(cleaned_text, tokenized_word, text_features)
        # topic = perform_topic_modeling(tokenized_sent, text_features)
        alert = alert_box(df, numeric_features, categoric_features, pearson_correlation)

        missing_barchart = generate_missing_values_barchart(missing_count)
        numeric_histrogram = generate_numeric_histrogram(df, numeric_features)
        numeric_boxplot = generate_numeric_boxplot(df, numeric_features)
        numeric_pairplot = generate_numeric_pairplot(df, numeric_features)
        numeric_heatmap = generate_numeric_heatmap(df, numeric_features)
        categoric_barchart = generate_categoric_barchart(df, categoric_features)
        categoric_piechart = generate_categoric_piechart(df, categoric_features)
        unigram, bigram, trigram = perform_word_ngram_analysis(cleaned_text, tokenized_word, text_features)

        text_wordcloud = generate_text_wordcloud(tokenized_word, text_features)
        unigram_barchart = generate_unigram_barchart(unigram, text_features)
        bigram_barchart = generate_bigram_barchart(bigram, text_features)
        trigram_barchart = generate_trigram_barchart(trigram, text_features)

        summary = generate_summary(dataset_statistics, numeric_statistics, pearson_correlation, chi_square, alert, text_statistics, unigram, bigram, trigram)

        return render_template(
            'statistics.html', alert=alert, dataset_statistics=dataset_statistics, missing_count=missing_count, 
            features_dropped=features_dropped, feature_groups=grouped_features,
            numeric_statistics=numeric_statistics, pearson_correlation=pearson_correlation,
            categoric_statistics=categoric_statistics, chi_square=chi_square,
            text_statistics=text_statistics, unigrams=unigram,bigrams=bigram, trigrams=trigram, 
            missing_barchart=missing_barchart,
            numeric_histrogram=numeric_histrogram, numeric_boxplot=numeric_boxplot,
            numeric_pairplot=numeric_pairplot, numeric_heatmap=numeric_heatmap, 
            categoric_barchart=categoric_barchart,categoric_piechart=categoric_piechart, 
            text_wordcloud=text_wordcloud, unigram_barchart=unigram_barchart,
            bigram_barchart=bigram_barchart, trigram_barchart=trigram_barchart, 
            summary=summary
        )
    except FileNotFoundError:
        flash("No dataset uploaded yet.", "danger")
        return redirect(url_for('home'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
