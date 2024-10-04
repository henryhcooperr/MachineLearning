import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import re
import tldextract
import matplotlib.pyplot as plt
import os
import argparse

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Phishing URL detection using a neural network.')
parser.add_argument('--train', action='store_true', help='Train the model using the dataset.')
parser.add_argument('--predict', type=str, help='Predict whether a given URL is phishing or legitimate.')
parser.add_argument('--plot', action='store_true', help='Plot the training and validation metrics if training.')

args = parser.parse_args()

# Check if the model exists
model_path = 'phishing_model.h5'
model_exists = os.path.exists(model_path)

# Load the model if it exists
if model_exists:
    model = tf.keras.models.load_model(model_path)
    print("Loaded saved model.")
else:
    # Train the model if requested or if no existing model is found
    if args.train or not model_exists:
        # Load your dataset into a DataFrame
        df = pd.read_csv('out.csv')
        
        # Display the initial shape of the DataFrame
        print(f"Initial DataFrame shape: {df.shape}")

        # Drop irrelevant features
        df = df.drop(columns=['url', 'source', 'whois_data', 'domain_age_days'])

        # Handle missing data by dropping rows with any missing values
        df = df.dropna()

        # Convert boolean columns to integers
        bool_cols = ['starts_with_ip', 'has_punycode', 'domain_has_digits', 'has_internal_links']
        df[bool_cols] = df[bool_cols].astype(int)

        # Encode the 'label' column
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])

        # Display the classes encoded in 'label'
        print(f"Classes in 'label': {le.classes_}")

        # Separate features and target variable
        X = df.drop(columns=['label'])
        y = df['label']

        # Standardize numerical features
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Build the neural network model
        model = Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=64,
            validation_data=(X_test, y_test)
        )

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

        # Save the model
        model.save(model_path)
        print(f"Model saved as '{model_path}'.")

        # Optional: Plot training & validation accuracy values if requested
        if args.plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Test')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Test')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.show()

# Function to extract features from a given URL
def extract_features(url):
    features = {}

    # Extract URL length
    features['url_length'] = len(url)

    # Extract if URL starts with an IP
    features['starts_with_ip'] = int(bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url)))

    # URL entropy - simplified version, calculated as character diversity in URL
    features['url_entropy'] = len(set(url)) / len(url) if len(url) > 0 else 0

    # Check for punycode
    features['has_punycode'] = int('xn--' in url)

    # Ratio of digits to letters in URL
    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    features['digit_letter_ratio'] = digit_count / letter_count if letter_count > 0 else 0

    # Extract count of specific characters
    features['dot_count'] = url.count('.')
    features['at_count'] = url.count('@')
    features['dash_count'] = url.count('-')

    # Extract top-level domain count
    tld_info = tldextract.extract(url)
    features['tld_count'] = len(tld_info.suffix.split('.')) if tld_info.suffix else 0

    # Check if domain has digits
    features['domain_has_digits'] = int(any(char.isdigit() for char in tld_info.domain))

    # Count the number of subdomains
    features['subdomain_count'] = len(tld_info.subdomain.split('.')) if tld_info.subdomain else 0

    # Calculate nan character entropy (simplified as non-alphanumeric characters diversity)
    non_alpha_numeric = re.sub(r'[a-zA-Z0-9]', '', url)
    features['nan_char_entropy'] = len(set(non_alpha_numeric)) / len(non_alpha_numeric) if len(non_alpha_numeric) > 0 else 0

    # Check if there are internal links (usually identified by `#` in URL)
    features['has_internal_links'] = int('#' in url)

    return pd.DataFrame([features])

# Function to predict if a URL is phishing or legitimate
def predict_url(url):
    # Extract features from the URL
    new_link_features = extract_features(url)

    # Standardize numerical features
    if not model_exists:
        new_link_features[num_cols] = scaler.transform(new_link_features[num_cols])

    # Make a prediction
    prediction = model.predict(new_link_features)
    label = 'phishing' if prediction[0] > 0.5 else 'legitimate'
    return label

# Predict based on command line argument
if args.predict:
    prediction_result = predict_url(args.predict)
    print(f'The link "{args.predict}" is predicted to be: {prediction_result}')
