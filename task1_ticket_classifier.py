# === STEP 1: Import Libraries ===
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === STEP 2: Download NLTK Tools ===
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# === STEP 3: Load Excel File ===
df = pd.read_excel("ai_dev_assignment_tickets_complex_1000.xls")
print(df.head())


# === STEP 4.5: Clean ticket_text and create clean_text column ===
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Apply cleaning to ticket_text column
df['clean_text'] = df['ticket_text'].apply(clean_text)



# === STEP 5: TF-IDF Vectorization ===
from sklearn.preprocessing import LabelEncoder

# Vectorize clean_text
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df['clean_text'])

# === STEP 6: Encode Labels ===
le_type = LabelEncoder()
le_urgency = LabelEncoder()

df['issue_type_enc'] = le_type.fit_transform(df['issue_type'])
df['urgency_level_enc'] = le_urgency.fit_transform(df['urgency_level'])

# === STEP 7: Train Classifier for issue_type ===
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, df['issue_type_enc'], test_size=0.2, random_state=42)
clf_issue = RandomForestClassifier()
clf_issue.fit(X_train1, y_train1)

print("Issue Type Classification Report:")
print(classification_report(y_test1, clf_issue.predict(X_test1)))

# === STEP 8: Train Classifier for urgency_level ===
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, df['urgency_level_enc'], test_size=0.2, random_state=42)
clf_urgency = RandomForestClassifier()
clf_urgency.fit(X_train2, y_train2)

print("Urgency Level Classification Report:")
print(classification_report(y_test2, clf_urgency.predict(X_test2)))


# === ENTITY EXTRACTION FUNCTION ===

def extract_entities(text):
    text = str(text).lower()
    
    product_keywords = ['smartwatch', 'vacuum', 'soundwave', 'cam', 'tv', 'printer', 'headphones', 'laptop']
    complaint_keywords = ['broken', 'error', 'damaged', 'not working', 'late', 'missing']
    
    found_products = [p for p in product_keywords if p in text]
    found_keywords = [k for k in complaint_keywords if k in text]
    
    dates = re.findall(r'\b\d{1,2} [A-Za-z]+ \d{4}\b', text)

    return {
        "products": list(set(found_products)),
        "keywords": list(set(found_keywords)),
        "dates": dates
    }


# === STEP 10: Final Analyze Function ===
def analyze_ticket(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    
    issue_pred = le_type.inverse_transform(clf_issue.predict(vec))[0]
    urgency_pred = le_urgency.inverse_transform(clf_urgency.predict(vec))[0]
    entities = extract_entities(text)
    
    return {
        "Predicted Issue Type": issue_pred,
        "Predicted Urgency Level": urgency_pred,
        "Extracted Entities": entities
    }

# === STEP 11: Test the Function ===
sample_text = "My SmartWatch stopped working and I got it late on 12 May 2025."
print(analyze_ticket(sample_text))
