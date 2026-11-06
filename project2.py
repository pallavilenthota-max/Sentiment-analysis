
# SOCIAL MEDIA SENTIMENT ANALYSIS
# Compare 4 Machine Learning Models

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# STEP 1: DATASET PATH
# --------------------------------------------------

SOCIAL_PATH = r"C:\Users\palla\OneDrive\Documents\ML PROJECTVDATSETS\SOCIAL MEDIA SENTIMENT ANALYSIS\sentimentdataset.csv"

OUTPUT_DIR = "sentiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# STEP 2: LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(SOCIAL_PATH)

df.columns = df.columns.astype(str).str.strip()

print("\nSOCIAL MEDIA SENTIMENT DATASET")
print("-" * 50)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# STEP 3: IDENTIFY REQUIRED COLUMNS
# --------------------------------------------------

# Find Text and Sentiment columns even if capitalization
# or spaces differ.

def normalize_column(name):
    return "".join(
        ch.lower() for ch in str(name)
        if ch.isalnum()
    )

column_map = {
    normalize_column(col): col
    for col in df.columns
}

text_candidates = ["text", "post", "tweet", "content"]
sentiment_candidates = [
    "sentiment", "label", "polarity", "emotion"
]

text_col = next(
    (
        column_map[name]
        for name in text_candidates
        if name in column_map
    ),
    None
)

sentiment_col = next(
    (
        column_map[name]
        for name in sentiment_candidates
        if name in column_map
    ),
    None
)

if text_col is None or sentiment_col is None:
    raise ValueError(
        "Could not identify the text and sentiment columns. "
        "Check the column names printed above and update "
        "text_col and sentiment_col manually."
    )

print("\nText column:", text_col)
print("Sentiment column:", sentiment_col)

# --------------------------------------------------
# STEP 4: CLEAN DATA
# --------------------------------------------------

df = df.dropna(
    subset=[text_col, sentiment_col]
).copy()

df[text_col] = df[text_col].astype(str).str.strip()
df[sentiment_col] = (
    df[sentiment_col].astype(str).str.strip()
)

df = df[
    (df[text_col] != "") &
    (df[sentiment_col] != "")
].copy()

# --------------------------------------------------
# STEP 5: GROUP SENTIMENT LABELS
# --------------------------------------------------
# This dataset may contain detailed emotion labels.
# The function groups common labels into Positive,
# Negative, and Neutral.
#
# Verify the mapping against your dataset before
# presenting the results.

positive_labels = {
    "positive", "joy", "happy", "happiness", "excited",
    "excitement", "gratitude", "grateful", "hopeful",
    "hope", "love", "optimism", "pride", "proud",
    "admiration", "enjoyment", "satisfaction",
    "inspiration", "inspired", "success", "celebration",
    "kindness", "affection", "amusement", "relief",
    "confidence", "contentment", "empowerment",
    "enthusiasm", "compassion", "blessed", "serenity",
    "playful", "determination", "awe", "fulfillment",
    "appreciation", "wonder", "wonderment", "harmony"
}

negative_labels = {
    "negative", "sad", "sadness", "anger", "angry",
    "fear", "fearful", "hate", "despair", "loneliness",
    "lonely", "frustration", "frustrated", "disgust",
    "jealousy", "jealous", "grief", "sorrow", "regret",
    "betrayal", "heartbreak", "anxiety", "helplessness",
    "suffering", "disappointment", "disappointed",
    "shame", "resentment", "exhaustion", "desperation",
    "melancholy", "bitterness", "devastated",
    "overwhelmed", "isolation", "heartache"
}

neutral_labels = {
    "neutral", "curiosity", "surprise", "anticipation",
    "reflection", "contemplation", "nostalgia",
    "indifference", "confusion", "ambivalence",
    "pensive", "numbness"
}


def group_sentiment(value):

    label = str(value).strip().lower()

    if label in positive_labels:
        return "Positive"

    if label in negative_labels:
        return "Negative"

    if label in neutral_labels:
        return "Neutral"

    # Handle labels containing recognizable keywords.
    if any(word in label for word in [
        "happy", "joy", "gratitude", "excite", "hope",
        "love", "optim", "proud", "admira", "enjoy",
        "satisf", "inspir", "success", "celebrat",
        "positive", "compassion", "kindness"
    ]):
        return "Positive"

    if any(word in label for word in [
        "sad", "anger", "angry", "fear", "hate",
        "despair", "lonely", "frustrat", "disgust",
        "jealous", "grief", "sorrow", "regret",
        "betray", "heartbreak", "anxiety", "helpless",
        "suffer", "negative", "disappoint", "shame",
        "resent", "exhaust", "devastat", "bitterness"
    ]):
        return "Negative"

    # Unrecognized labels are assigned to Neutral.
    # Check these labels before using the results.
    return "Neutral"


print("\nOriginal sentiment labels:")
print(df[sentiment_col].value_counts())

df["Sentiment_Group"] = df[sentiment_col].apply(
    group_sentiment
)

print("\nGrouped sentiment distribution:")
print(df["Sentiment_Group"].value_counts())

# --------------------------------------------------
# STEP 6: PREPARE TRAINING AND TESTING DATA
# --------------------------------------------------

X = df[text_col]
y = df["Sentiment_Group"]

if y.nunique() < 2:
    raise ValueError(
        "Only one sentiment class was detected. "
        "Check the sentiment mapping."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "A sentiment class has fewer than two records. "
        "Check the label mapping or dataset."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# STEP 7: DEFINE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "Multinomial Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )),
        ("model", MultinomialNB())
    ]),

    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )),
        ("model", LinearSVC(
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("model", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])
}

# --------------------------------------------------
# STEP 8: TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []
predictions = {}

for name, model in models.items():

    print("\nTraining model:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    results.append({
        "Model": name,

        "Accuracy": accuracy_score(
            y_test, y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1-score": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    })

# --------------------------------------------------
# STEP 9: DISPLAY MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("SOCIAL MEDIA SENTIMENT MODEL COMPARISON")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_model_comparison.csv"
    ),
    index=False
)

# --------------------------------------------------
# STEP 10: MODEL COMPARISON GRAPH
# --------------------------------------------------

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-score"]
].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1)
)

plt.title("Social Media Sentiment - Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Evaluation Score")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_model_comparison.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 11: CONFUSION MATRIX
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_predictions = predictions[best_model_name]

class_names = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=class_names
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Sentiment Confusion Matrix - " + best_model_name
)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 12: SENTIMENT DISTRIBUTION GRAPH
# --------------------------------------------------

df["Sentiment_Group"].value_counts().reindex(
    class_names,
    fill_value=0
).plot(
    kind="bar",
    figsize=(7, 4)
)

plt.title("Social Media Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_distribution.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 13: CLASSIFICATION REPORT
# --------------------------------------------------

print("\nBEST MODEL BY WEIGHTED F1-SCORE:")
print(best_model_name)

print("\nCLASSIFICATION REPORT:")

print(classification_report(
    y_test,
    best_predictions,
    labels=class_names,
    zero_division=0
))

# --------------------------------------------------
# STEP 14: FINAL OUTPUT
# --------------------------------------------------

print("\nPROJECT COMPLETED SUCCESSFULLY")

print(
    "\nResults and graphs saved in:",
    os.path.abspath(OUTPUT_DIR)
)

