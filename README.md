Cyber Estate – Property Price & ROI Prediction App

This project is a Machine Learning–based web application that predicts property prices and evaluates investment ROI using features such as city, property type, built-up area, property age, BHK, furnishing, and parking. The model is deployed using Streamlit and leverages saved preprocessing pipelines (.pkl files) to ensure consistent and reliable predictions during inference.

Features

Predicts property prices based on location and property attributes

Classifies investment outlook as High ROI or Low–Medium ROI

Uses trained ML models with saved encoders and scalers

Clean, premium, and interactive UI built with Streamlit

Context-aware inputs (e.g., BHK and furnishing ignored for plots)

Project Structure
├── property_price_app.py
├── price_model.pkl
├── price_scaler.pkl
├── price_column_transformer.pkl
├── price_feature_names.pkl
├── roi_model.pkl
├── roi_scaler.pkl
├── roi_column_transformer.pkl
├── roi_feature_names.pkl

Use Case

Ideal for real estate analytics, ML deployment practice, and data science portfolios.

Contribution

Pull requests and suggestions are welcome.
