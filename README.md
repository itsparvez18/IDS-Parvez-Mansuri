# 🛡️ AI Network Intrusion Detection System

An AI-powered Network Intrusion Detection System (IDS) designed to detect and classify malicious **DDoS network traffic** using Machine Learning.

The project uses network-flow data, performs preprocessing and feature preparation, trains multiple Machine Learning models, evaluates their performance, and provides a Streamlit-based dashboard for traffic analysis and intrusion detection.

---

## 📌 Project Overview

Traditional network monitoring systems can generate large amounts of traffic data that are difficult to analyze manually.

This project applies Machine Learning to network-flow data to automatically classify traffic into:

- 🟢 **BENIGN** — Normal network traffic
- 🔴 **DDoS** — Distributed Denial-of-Service traffic

The trained Random Forest model is integrated into a Streamlit dashboard where users can analyze network traffic and perform predictions using CSV files.

The project also includes a **real-time monitoring simulation** that processes network-flow records in batches and generates DDoS alerts.

---

## 🎯 Objectives

- Detect malicious DDoS network traffic using Machine Learning.
- Preprocess and clean network-flow datasets.
- Compare multiple Machine Learning algorithms.
- Evaluate classification performance.
- Build an interactive cybersecurity dashboard.
- Provide CSV-based network traffic prediction.
- Generate DDoS detection alerts.
- Maintain detection logs for monitoring.

---

## 🧠 Machine Learning Models

Three Machine Learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Random Forest model was selected as the final detection model based on the evaluation results.

---

## 📊 Dataset

The project uses network-flow data from the **CIC-IDS2017** dataset.

The specific dataset used for this project contains:

- **225,745** original network-flow records
- **85** original columns
- DDoS and BENIGN traffic
- Network-flow based features

After preprocessing:

- **225,619** records
- **80** numerical features
- **97,592** BENIGN records
- **128,027** DDoS records

### Preprocessing

The preprocessing pipeline includes:

- Removing unnecessary identifier columns
- Handling missing values
- Handling infinite values
- Removing duplicate records
- Converting labels into numerical form
- Selecting numerical network-flow features

Label encoding:

```text
BENIGN → 0
DDoS   → 1
