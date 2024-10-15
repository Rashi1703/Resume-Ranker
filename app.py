from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources from NLTK
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load CSV file for model training
csv_file_path = 'resume_job_matching.csv'
if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
else:
    raise FileNotFoundError(f"The file {csv_file_path} does not exist.")

# Data Cleaning Function
def clean_text(text: str) -> str:
    """Clean the input text by removing special characters, extra spaces, and lowercasing.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s.,-]', '', text)  # Remove special characters except for some punctuation
    return text.lower()  # Convert to lowercase

# Apply cleaning to the DataFrame
df['Resume_Text'] = df['Resume_Text'].apply(clean_text)
df['Job_Description'] = df['Job_Description'].apply(clean_text)

# Preprocessing: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_resumes = vectorizer.fit_transform(df['Resume_Text'])
X_jobs = vectorizer.transform(df['Job_Description'])
X = (X_resumes + X_jobs) / 2
y = df['Matching_score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models globally
rf_model = RandomForestRegressor().fit(X_train, y_train)
lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
svm_model = SVC(probability=True).fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
mlp_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500).fit(X_train, y_train)

def preprocess_text(text: str) -> str:
    """Preprocess the input text by removing stop words and lemmatizing.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W+', ' ', text)
    # Tokenize the text
    words = text.split()
    # Remove stop words and lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract and preprocess text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The preprocessed text extracted from the PDF.
    """
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return preprocess_text(text)

def extract_text_from_docx(file_path: str) -> str:
    """Extract and preprocess text from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The preprocessed text extracted from the DOCX.
    """
    return preprocess_text(docx2txt.process(file_path))

def extract_text_from_txt(file_path: str) -> str:
    """Extract and preprocess text from a TXT file.

    Args:
        file_path (str): The path to the TXT file.

    Returns:
        str: The preprocessed text extracted from the TXT file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return preprocess_text(file.read())

def extract_text(file_path: str) -> str:
    """Determine file type and extract text accordingly.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The preprocessed text extracted from the file, or an empty string if unsupported.
    """
    # Determine file type and extract text accordingly
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

@app.route("/")
def matchresume():
    """Render the main form for uploading resumes and job description.

    Returns:
        str: The rendered HTML template for the resume matching page.
    """
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    """Handle the matching of resumes to job descriptions.

    Returns:
        str: The rendered HTML template for the resume matching page with results.
    """
    if request.method == 'POST':
        # Get the list of uploaded resumes
        resume_files = request.files.getlist('resumes')
        job_description = request.form['job_description']

        # Check if resumes and job description are provided
        if not resume_files or not job_description:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        resumes = []
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)  # Save the resume file
            resumes.append(extract_text(filename))  # Extract and preprocess text

        # Vectorize the job description and resumes
        vectors = vectorizer.transform(resumes)
        job_vector = vectorizer.transform([job_description])

        # Calculate cosine similarities between job description and resumes
        similarities = cosine_similarity(job_vector, vectors)[0]

        # Prepare DataFrame for predictions
        resume_job_df = pd.DataFrame({
            'Resume_Text': resumes,
            'Job_Description': [job_description] * len(resumes)
        })

        X_resumes = vectorizer.transform(resume_job_df['Resume_Text'])
        X_jobs = vectorizer.transform(resume_job_df['Job_Description'])
        X = (X_resumes + X_jobs) / 2  # Combine resume and job description

        # Sort the resumes by similarity score
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]

        # Make predictions
        predictions = {
            'Random Forest': rf_model.predict(X),
            'Logistic Regression': lr_model.predict_proba(X)[:, 1] * 100,
            'SVM': svm_model.predict_proba(X)[:, 1] * 100,
            'KNN': knn_model.predict_proba(X)[:, 1] * 100,
            'MLP': mlp_model.predict_proba(X)[:, 1] * 100,
            'similarity_scores': [round(similarities[i], 2) * 750 for i in top_indices]
        }

        # Aggregate results
        combined_predictions = np.array(list(predictions.values()))
        similarity_score = np.mean(combined_predictions, axis=0)

        # Render the result page showing top matching resumes and their similarity scores
        return render_template(
            'matchresume.html', 
            message="Top matching resumes:", 
            top_resumes=top_resumes, 
            similarity_scores=similarity_score[top_indices].tolist()  
        )

    # If method is not POST, render the main page again
    return render_template('matchresume.html')

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Run the Flask application in debug mode
    app.run(debug=True)
