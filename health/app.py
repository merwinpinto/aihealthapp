from flask import Flask, request, render_template, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables
user_symptoms = []
faq_df = None
questions_list = []
tokenizer = None
model = None
faq_embeddings = None
faq_clusters = None
cluster_centers = None
category_model = None
subcategory_model = None
all_symptoms = set()
label_encoder_category = None
label_encoder_subcategory = None

def initialize_models():
    """Initialize all models and data"""
    global faq_df, questions_list, tokenizer, model, faq_embeddings, faq_clusters, cluster_centers
    global category_model, subcategory_model, all_symptoms, label_encoder_category, label_encoder_subcategory

    try:
        # Load the FAQ dataset
        if os.path.exists('faq.csv'):
            faq_df = pd.read_csv('faq.csv')
            questions_list = faq_df['Question'].tolist()
        else:
            # Create a sample FAQ dataset
            sample_data = {
                'Question': [
                    'What are the symptoms of COVID-19?',
                    'How to prevent flu?',
                    'What causes fever?',
                    'How to treat headache?',
                    'What are respiratory symptoms?',
                    'What causes conjunctivitis?',
                    'How to treat eye infections like trachoma?'
                ],
                'Answer': [
                    'COVID-19 symptoms include fever, cough, difficulty breathing, loss of taste and smell.',
                    'Get vaccinated, wash hands frequently, avoid crowded places.',
                    'Fever can be caused by infections, inflammation, or other medical conditions.',
                    'Rest, hydration, and over-the-counter pain relievers can help with headaches.',
                    'Respiratory symptoms include cough, difficulty breathing, sore throat, and wheezing.',
                    'Conjunctivitis can be caused by viruses, bacteria, or allergies, leading to redness and itching.',
                    'Trachoma requires antibiotics like azithromycin; consult a healthcare provider.'
                ]
            }
            faq_df = pd.DataFrame(sample_data)
            faq_df.to_csv('faq.csv', index=False)
            questions_list = faq_df['Question'].tolist()

        # Load pre-trained model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')

        # Get embeddings for FAQ questions
        faq_embeddings = get_embeddings(faq_df['Question'].tolist())

        # Cluster FAQ embeddings
        faq_embeddings_np = faq_embeddings.detach().numpy()
        faq_clusters, cluster_centers = cluster_faqs(faq_embeddings_np)
        faq_df['Cluster'] = faq_clusters

        # Initialize ML models for disease prediction
        setup_disease_prediction_models()

    except Exception as e:
        print(f"Error initializing models: {e}")
        faq_df = pd.DataFrame({'Question': ['Sample question'], 'Answer': ['Sample answer']})
        questions_list = ['Sample question']

def get_embeddings(texts):
    """Get embeddings from BERT model"""
    try:
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return torch.zeros((len(texts), 768))

def cluster_faqs(embeddings, n_clusters=5):
    """Cluster FAQ questions using K-Means"""
    try:
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=0)
        kmeans.fit(embeddings)
        return kmeans.labels_, kmeans.cluster_centers_
    except Exception as e:
        print(f"Error clustering FAQs: {e}")
        return np.zeros(len(embeddings)), np.zeros((min(n_clusters, len(embeddings)), embeddings.shape[1]))

def get_category_weights():
    """Get category weights for disease prediction"""
    return {
        'Respiratory Infections': {
            'Cough': 0.217, 'Fever': 0.173, 'Difficulty-in-Breathing': 0.130, 'Sore-Throat': 0.130,
            'Fatigue': 0.087, 'Wheezing': 0.087, 'Nasal-Congestion': 0.087, 'Runny-Nose': 0.087,
            'Loss of Taste': 0.044, 'Loss of Smell': 0.044, 'Sneezing': 0.044, 'Chills': 0.044,
            'Headache': 0.044
        },
        'Skin Infection': {
            'Rash': 0.4, 'Fever': 0.2, 'Body-Pain': 0.15, 'Fatigue': 0.1, 'Sore-Throat': 0.15
        },
        'Gastrointestinal Infections': {
            'Diarrhea': 0.35, 'Vomiting': 0.35, 'Abdominal-Pain': 0.2, 'Nausea': 0.05, 'Loss-of-Appetite': 0.05
        },
        'Systemic Infections': {
            'Loss of Taste': 0.25, 'Loss of Smell': 0.25, 'Fever': 0.20,
            'Chills': 0.1, 'Fatigue': 0.05, 'Body-Pain': 0.15
        },
        'Genital Infection': {
            'Genital-Lesions': 0.410, 'Painful-Urination': 0.310, 'Abnormal-Discharge': 0.150,
            'Swollen-Lymph-Nodes': 0.050, 'Fever': 0.050, 'Burning-Sensation': 0.030
        },
        'Oral Infection': {
            'Oral-Lesions': 0.35, 'Difficulty-Swallowing': 0.3, 'Sore-Throat': 0.15,
            'Swollen-Gums': 0.1, 'Mouth-Ulcers': 0.05, 'Bad-Breath': 0.05
        },
        'Arboviral Infections (Insect bite)': {
            'Rash': 0.310, 'Joint-Pain': 0.250, 'Fever': 0.160, 'Muscle-Pain': 0.110,
            'Fatigue': 0.110, 'Eye-Redness': 0.060  # Replaced Conjunctivitis
        },
        'Hemorrhagic Fevers': {
            'Hemorrhagic-Rash': 0.3, 'Blood-in-Vomit': 0.2, 'Blood-in-Stool': 0.2,
            'Severe-Headache': 0.1, 'Jaundice': 0.05, 'Muscle-Pain': 0.05,
            'Fatigue': 0.05, 'Gum-Bleeding': 0.05, 'Nosebleeds': 0.05
        },
        'Eye Infection': {
            'Redness': 0.300, 'Itching': 0.250, 'Discharge': 0.200, 'Pain': 0.150,
            'Blurred-Vision': 0.050, 'Sensitivity-to-Light': 0.050
        }
    }

def get_sub_category_weights():
    """Get sub-category weights for disease prediction based on recent WHO health records"""
    return {
        'Respiratory Infections': {
            'Common Cold': {
                'Nasal-Congestion': 0.350, 'Runny-Nose': 0.300, 'Sore-Throat': 0.050,
                'Sneezing': 0.100, 'Cough': 0.150, 'Fever': 0.050
            },
            'Flu (Influenza)': {
                'Fever': 0.350, 'Chills': 0.250, 'Muscle-Pain': 0.100,
                'Cough': 0.200, 'Nasal-Congestion': 0.050, 'Runny-Nose': 0.050
            },
            'Covid-19': {
                'Fever': 0.300, 'Fatigue': 0.250, 'Cough': 0.200,
                'Difficulty-in-Breathing': 0.150, 'Loss of Taste': 0.050, 'Loss of Smell': 0.050
            },
            'Pneumonia': {
                'Cough': 0.300, 'Fever': 0.250, 'Difficulty-in-Breathing': 0.200,
                'Fatigue': 0.150, 'Chills': 0.100
            },
            'Tuberculosis': {
                'Cough': 0.350, 'Fever': 0.200, 'Fatigue': 0.200,
                'Difficulty-in-Breathing': 0.150, 'Chills': 0.100
            }
        },
        'Skin Infection': {
            'Bacterial Skin Infection': {
                'Rash': 0.400, 'Fever': 0.250, 'Body-Pain': 0.200, 'Fatigue': 0.150
            },
            'Fungal Skin Infection': {
                'Rash': 0.500, 'Sore-Throat': 0.200, 'Fatigue': 0.150, 'Body-Pain': 0.150
            },
            'Viral Skin Infection': {
                'Rash': 0.450, 'Body-Pain': 0.250, 'Fever': 0.150, 'Fatigue': 0.150
            },
            'Paronychia (Nail Infection)': {
                'Swelling': 0.350, 'Pain': 0.350, 'Redness': 0.300
            }
        },
        'Gastrointestinal Infections': {
            'Viral Gastroenteritis': {
                'Vomiting': 0.400, 'Diarrhea': 0.350, 'Nausea': 0.150, 'Abdominal-Pain': 0.100
            },
            'Bacterial Gastroenteritis': {
                'Diarrhea': 0.400, 'Abdominal-Pain': 0.250, 'Vomiting': 0.200, 'Fever': 0.150
            },
            'Parasitic Infection': {
                'Diarrhea': 0.350, 'Abdominal-Pain': 0.300, 'Loss-of-Appetite': 0.200, 'Nausea': 0.150
            },
            'Cholera': {
                'Diarrhea': 0.450, 'Vomiting': 0.300, 'Abdominal-Pain': 0.150, 'Nausea': 0.100
            }
        },
        'Systemic Infections': {
            'Viral Systemic Infection': {
                'Fatigue': 0.300, 'Fever': 0.250, 'Body-Pain': 0.250, 'Loss of Taste': 0.200
            },
            'Anosmia-Associated Infection': {
                'Loss of Taste': 0.350, 'Loss of Smell': 0.350, 'Fever': 0.150, 'Fatigue': 0.150
            }
        },
        'Genital Infection': {
            'Bacterial Vaginosis': {
                'Abnormal-Discharge': 0.400, 'Burning-Sensation': 0.250, 'Painful-Urination': 0.200, 'Fever': 0.150
            },
            'Genital Herpes': {
                'Genital-Lesions': 0.450, 'Painful-Urination': 0.250, 'Burning-Sensation': 0.200, 'Fever': 0.100
            },
            'Gonorrhea': {
                'Painful-Urination': 0.350, 'Abnormal-Discharge': 0.300, 'Genital-Lesions': 0.200, 'Fever': 0.150
            },
            'Urinary Tract Infection':{
                 'Fever': 0.500,'Itching':0.500
            }
        },
        'Oral Infection': {
            'Poor Oral Hygiene': {
                'Swollen-Gums': 0.350, 'Bad-Breath': 0.250, 'Mouth-Ulcers': 0.400
            },
            'Oral Thrush': {
                'Oral-Lesions': 0.400, 'Difficulty-Swallowing': 0.250, 'Sore-Throat': 0.200, 'Mouth-Ulcers': 0.150
            }
        },
        'Arboviral Infections (Insect bite)': {
            'Dengue Fever': {
                'Fever': 0.300, 'Joint-Pain': 0.250, 'Muscle-Pain': 0.200, 'Rash': 0.150, 'Fatigue': 0.100
            },
            'Zika Virus': {
                'Rash': 0.350, 'Fever': 0.200, 'Joint-Pain': 0.200, 'Eye-Redness': 0.150, 'Fatigue': 0.100
            },
            'Chikungunya': {
                'Joint-Pain': 0.350, 'Fever': 0.250, 'Rash': 0.200, 'Muscle-Pain': 0.150, 'Fatigue': 0.050
            }
        },
        'Hemorrhagic Fevers': {
            'Ebola': {
                'Fever': 0.300, 'Blood-in-Vomit': 0.250, 'Blood-in-Stool': 0.200, 'Fatigue': 0.150, 'Muscle-Pain': 0.100
            },
            'Lassa Fever': {
                'Fever': 0.350, 'Severe-Headache': 0.200, 'Fatigue': 0.200, 'Nosebleeds': 0.150, 'Gum-Bleeding': 0.100
            },
            'Crimean-Congo Haemorrhagic Fever': {
                'Fever': 0.300, 'Hemorrhagic-Rash': 0.250, 'Severe-Headache': 0.200, 'Nosebleeds': 0.150, 'Fatigue': 0.100
            }
        },
        'Eye Infection': {
            'Viral infection': {
                'Redness': 0.350, 'Discharge': 0.300, 'Itching': 0.250, 'Pain': 0.100
            },
            'Bacterial infection': {
                'Discharge': 0.350, 'Redness': 0.300, 'Pain': 0.200, 'Itching': 0.150
            },
            'Allergic Conjunctivitis': {
                'Itching': 0.400, 'Redness': 0.350, 'Discharge': 0.150, 'Sensitivity-to-Light': 0.100
            }
        }
    }

def generate_synthetic_data(category_weights, sub_category_weights, num_samples=2000):
    """Generate synthetic data for training with balanced sub-categories"""
    data = []
    total_subcategories = sum(len(subcats) for subcats in sub_category_weights.values())
    samples_per_subcategory = max(1, num_samples // max(1, total_subcategories))

    for category, symptoms_weights in category_weights.items():
        sub_categories = sub_category_weights.get(category, {})
        if sub_categories:
            for sub_category, sub_symptoms_weights in sub_categories.items():
                for _ in range(samples_per_subcategory):
                    # Generate category-level symptoms
                    symptoms = []
                    for symptom, weight in symptoms_weights.items():
                        if random.random() < weight:
                            symptoms.append(symptom)
                    # Generate sub-category symptoms
                    sub_symptoms = []
                    for symptom, weight in sub_symptoms_weights.items():
                        if random.random() < weight:
                            sub_symptoms.append(symptom)
                    # Combine and deduplicate symptoms
                    symptoms = list(set(symptoms + sub_symptoms))
                    data.append((category, sub_category, symptoms))
        else:
            # For categories without sub-categories
            for _ in range(samples_per_subcategory):
                symptoms = []
                for symptom, weight in symptoms_weights.items():
                    if random.random() < weight:
                        symptoms.append(symptom)
                data.append((category, "Unknown", symptoms))

    return pd.DataFrame(data, columns=['Category', 'Subcategory', 'Symptoms'])

def setup_disease_prediction_models():
    """Setup machine learning models for disease prediction"""
    global category_model, subcategory_model, all_symptoms, label_encoder_category, label_encoder_subcategory

    try:
        category_weights = get_category_weights()
        sub_category_weights = get_sub_category_weights()

        # Generate synthetic data
        df = generate_synthetic_data(category_weights, sub_category_weights)

        # Prepare features
        all_symptoms = set(symptom for symptoms_list in df['Symptoms'] for symptom in symptoms_list)
        for symptom in all_symptoms:
            df[symptom] = df['Symptoms'].apply(lambda x: 1 if symptom in x else 0)
        df.drop(columns=['Symptoms'], inplace=True)

        # Encode labels
        label_encoder_category = LabelEncoder()
        label_encoder_subcategory = LabelEncoder()
        df['CategoryEncoded'] = label_encoder_category.fit_transform(df['Category'])
        df['SubcategoryEncoded'] = label_encoder_subcategory.fit_transform(df['Subcategory'])

        # Train models
        symptom_columns = list(all_symptoms)
        X = df[symptom_columns]
        y_category = df['CategoryEncoded']
        y_subcategory = df['SubcategoryEncoded']

        X_train, X_test, y_train_category, y_test_category = train_test_split(X, y_category, test_size=0.2, random_state=42)
        X_train, X_test, y_train_subcategory, y_test_subcategory = train_test_split(X, y_subcategory, test_size=0.2, random_state=42)

        category_model = RandomForestClassifier(n_estimators=100, random_state=42)
        category_model.fit(X_train, y_train_category)

        subcategory_model = RandomForestClassifier(n_estimators=100, random_state=42)
        subcategory_model.fit(X_train, y_train_subcategory)

    except Exception as e:
        print(f"Error setting up disease prediction models: {e}")

def answer_question(user_query):
    """Answer a question using FAQ matching"""
    try:
        if faq_df is None or len(faq_df) == 0:
            return "Sorry, no FAQ data available.", "No question available"

        query_embedding = get_embeddings([user_query])
        query_embedding_np = query_embedding.detach().numpy()

        similarities = cosine_similarity(query_embedding_np, cluster_centers)
        closest_cluster_idx = similarities.argmax()

        cluster_faqs = faq_df[faq_df['Cluster'] == closest_cluster_idx]
        if len(cluster_faqs) == 0:
            cluster_faqs = faq_df

        cluster_faq_embeddings = faq_embeddings[faq_df['Cluster'] == closest_cluster_idx]
        similarities_within_cluster = cosine_similarity(query_embedding, cluster_faq_embeddings)
        most_similar_idx_within_cluster = similarities_within_cluster.argmax()

        most_similar_question = cluster_faqs.iloc[most_similar_idx_within_cluster]['Question']
        answer = cluster_faqs.iloc[most_similar_idx_within_cluster]['Answer']

        return answer, most_similar_question
    except Exception as e:
        print(f"Error answering question: {e}")
        return "Sorry, I couldn't process your question.", "Error occurred"

# Initialize models when the app starts
initialize_models()

# Flask routes
@app.route('/')
def home():
    return render_template('index.html', questions=questions_list)

@app.route('/check-symptoms', methods=['GET', 'POST'])
def check_symptoms():
    all_symptoms = {
        'Respiratory Symptoms': [
            'Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Wheezing', 'Runny-Nose',
            'Nasal-Congestion', 'Sneezing'
        ],
        'Fever and Systemic Symptoms': [
            'Fever', 'Fatigue', 'Chills', 'Loss of Taste', 'Loss of Smell'
        ],
        'Pain and Musculoskeletal Symptoms': [
            'Body-Pain', 'Joint-Pain', 'Muscle-Pain', 'Headache', 'Severe-Headache'
        ],
        'Skin Symptoms': [
            'Rash', 'Hemorrhagic-Rash'
        ],
        'Digestive Symptoms': [
            'Diarrhea', 'Vomiting', 'Abdominal-Pain', 'Nausea', 'Loss-of-Appetite',
            'Blood-in-Vomit', 'Blood-in-Stool', 'Jaundice'
        ],
        'Oral Symptoms': [
            'Oral-Lesions', 'Difficulty-Swallowing', 'Swollen-Gums', 'Mouth-Ulcers', 'Bad-Breath', 'Gum-Bleeding'
        ],
        'Genital Symptoms': [
            'Genital-Lesions', 'Painful-Urination', 'Abnormal-Discharge', 'Swollen-Lymph-Nodes', 'Burning-Sensation'
        ],
        'Eye Symptoms': [
            'Redness', 'Itching', 'Discharge', 'Pain', 'Blurred-Vision', 'Sensitivity-to-Light'
        ],
        'Bleeding Symptoms': [
            'Nosebleeds', 'Gum-Bleeding', 'Blood-in-Vomit', 'Blood-in-Stool', 'Hemorrhagic-Rash'
        ]
    }
    return render_template('check-symptoms.html', symptoms=all_symptoms)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_query = request.form.get('query', '')
    if not user_query.strip():
        return render_template('faq.html', questions=questions_list[:5])

    answer, faq = answer_question(user_query)
    return render_template('faq.html', user_query=user_query, faq=faq, answer=answer, questions=questions_list[:10])

@app.route('/feedback', methods=['POST'])
def feedback():
    user_query = request.form.get('user_query', '')
    faq = request.form.get('faq', '')
    answer = request.form.get('answer', '')
    feedback_text = request.form.get('feedback', '')

    try:
        with open('feedback.txt', 'a') as f:
            f.write(f"Query: {user_query}, FAQ: {faq}, Answer: {answer}, Feedback: {feedback_text}\n")
    except Exception as e:
        print(f"Error saving feedback: {e}")

    return render_template('faq.html', feedback_received=True, user_query=user_query, faq=faq, answer=answer, questions=questions_list)

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')

@app.route('/disease-info')
def disease_info():
    return render_template('disease-info.html')

@app.route('/about-our-ai-tool')
def about_tool():
    return render_template('about-our-ai-tool.html')

@app.route('/contact-us')
def contact_us():
    return render_template('contact-us.html')

@app.route('/faq')
def faq_page():
    return render_template('faq.html', questions=questions_list[:5])

@app.route('/results', methods=['GET', 'POST'])
def results():
    try:
        global user_symptoms, category_model, subcategory_model, all_symptoms, label_encoder_category, label_encoder_subcategory
        if not all([category_model, subcategory_model, label_encoder_category, label_encoder_subcategory]):
            return render_template('results.html', error="Models not initialized. Please try again later.")

        selected_symptoms = request.args.getlist('symptoms[]')

        if not selected_symptoms:
            return render_template('results.html', error="No symptoms selected. Please go back and select symptoms.")

        if len(selected_symptoms) <= 1:
            return render_template('results.html',
                                   error="At least 2 symptoms are required for reliable analysis.",
                                   symptoms=selected_symptoms,
                                   warning="Too few symptoms selected. Please provide at least 2.")

        user_symptoms = selected_symptoms
        category_weights = get_category_weights()
        sub_category_weights = get_sub_category_weights()

        # Prepare input vector
        symptom_vector = np.zeros(len(all_symptoms))
        for i, symptom in enumerate(all_symptoms):
            if symptom in selected_symptoms:
                symptom_vector[i] = 1
        symptom_vector = symptom_vector.reshape(1, -1)

        # ML prediction
        category_pred = category_model.predict(symptom_vector)[0]
        category_proba = category_model.predict_proba(symptom_vector)[0]
        category_name = label_encoder_category.inverse_transform([category_pred])[0]
        category_confidence = max(category_proba)

        # Weighted scoring
        def calculate_scores(symptoms, weights):
            total_weight = sum(weights.values())
            match_score = sum(weights.get(symptom, 0) for symptom in symptoms)
            return match_score / total_weight if total_weight > 0 else 0

        def find_best_category(symptoms, weights, threshold=0.05):
            best_cat = "Unknown"
            best_score = 0
            all_scores = {}
            for cat, cat_weights in weights.items():
                score = calculate_scores(symptoms, cat_weights)
                all_scores[cat] = score
                if score > best_score:
                    best_cat = cat
                    best_score = score
            return best_cat if best_score >= threshold else "Unknown", best_score, all_scores

        weighted_category, weighted_score, category_scores = find_best_category(user_symptoms, category_weights)

        # Decision logic
        final_category = category_name
        final_confidence = category_confidence

        if category_confidence < 0.6 and weighted_score >= 0.7:
            final_category = weighted_category
            final_confidence = weighted_score

        if final_confidence < 0.5:
            final_category = "Unknown"

        # Subcategory logic
        subcategory_name = "Unknown infection"
        subcategory_confidence = 0
        if final_category in sub_category_weights:
            best_match = None
            best_overlap = 0
            for subcat, weights in sub_category_weights[final_category].items():
                overlap = sum(1 for s in weights if s in user_symptoms)
                if overlap / len(weights) >= 0.5 and overlap > best_overlap:
                    best_match = subcat
                    best_overlap = overlap
            if best_match:
                sub_pred = subcategory_model.predict(symptom_vector)[0]
                sub_proba = subcategory_model.predict_proba(symptom_vector)[0]
                subcategory_name = label_encoder_subcategory.inverse_transform([sub_pred])[0]
                subcategory_confidence = max(sub_proba)

        result = {
            "Predictions": {
                "ML Category Prediction": category_name,
                "ML Subcategory Prediction": subcategory_name,
                "Weighted Category Prediction": weighted_category,
                "Final Category Prediction": final_category
            },
            "Scores": {
                "ML Category Confidence": round(category_confidence, 3),
                "ML Subcategory Confidence": round(subcategory_confidence, 3),
                "Weighted Category Score": round(weighted_score, 3),
                "Confidence": "High" if final_confidence > 0.8 else "Medium" if final_confidence > 0.5 else "Low",
                "All Category Scores": {k: round(v, 3) for k, v in category_scores.items()}
            }
        }

        warning = ""
        if len(selected_symptoms) <= 1:
            warning = "Too few symptoms for reliable prediction. Add more if possible."
        elif final_category == "Unknown":
            warning = "Unable to confidently identify a disease category. Please consider consulting a healthcare provider."

        return render_template('results.html', res=result, symptoms=selected_symptoms, warning=warning)

    except Exception as e:
        print(f"Error in results: {e}")
        return render_template('results.html', error="An error occurred during analysis. Please try again.")

# import os, psutil

# process = psutil.Process(os.getpid())
# print("CPU %:", process.cpu_percent(interval=1.0))
# print("RAM MB:", process.memory_info().rss / (1024 * 1024))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
