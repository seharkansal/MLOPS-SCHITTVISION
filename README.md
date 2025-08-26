## SchittVision – Emotionally Adaptive Chatbot with Schitt's Creek Characters

### Overview

SchittVision is an end-to-end MLOps project aimed at building a production-grade chatbot capable of delivering emotionally adaptive responses in the style of Schitt’s Creek characters. It combines NLP, deep learning, and MLOps best practices to create a system that not only understands user emotions but also responds in a contextually relevant and character-driven manner.

### Key Features

* **Emotion Detection**: Experimented with BERT from scratch, later transitioned to pretrained BERT for more effective emotion classification.
* **Response Generation**: Fine-tuned GPT-2 to generate character-driven dialogue, incorporating emotion cues and context.
* **Data Pipeline**:

  * Web scraped dialogues from Schitt’s Creek scripts.
  * Cleaned and preprocessed data to create input-target pairs with context windows.
  * Added custom tokens for characters, emotions, and responses.
* **Model Training & Tracking**:

  * Tokenization and training of BERT and GPT-2 models.
  * Implemented **DVC** for dataset and model versioning.
  * Used **MLflow** for experiment tracking and model registry.
* **API & Deployment**:

  * Built a REST API using **Flask** to serve the models.
  * Wrote unit tests for models and API endpoints, integrated into CI/CD pipelines.
  * Containerized the application with **Docker**, integrated into CI/CD using **GitHub Actions**.
  * Deployed on a **self-hosted AWS EC2 runner**, with experiments in **local Kubernetes orchestration**.
* **Monitoring**: Currently adding **Prometheus and Grafana** to track model performance and trigger retraining when metrics degrade.

### Architecture Diagram

```
[Web Scraper] -> [Data Preprocessing & Tokenization] -> [BERT (Emotion Detection)]
[GPT-2 (Dialogue Generation)] -> [Flask API] -> [Docker Container]
[CI/CD Pipeline: GitHub Actions -> EC2 Runner] -> [Kubernetes (local experiments)]
[Monitoring: Prometheus + Grafana]
```

### Lessons Learned

* Starting with BERT from scratch was educational but resource-heavy; pretrained models delivered better results faster.
* Adding emotions to the dataset directly was less effective than separating emotion detection and response generation.
* Special tokens improved GPT-2’s understanding of context and character identity.
* Local Kubernetes was a good learning step, but production-ready orchestration requires more setup.
* CI/CD and containerization made deployments reproducible and scalable.

### Repository Highlights

* **/data\_pipeline**: Scripts for scraping, cleaning, and preparing data.
* **/models**: Training scripts for BERT and GPT-2.
* **/api**: Flask endpoints serving emotion and dialogue predictions.
* **/ci\_cd**: GitHub Actions workflows for building, testing, and deployment.
* **/monitoring**: Initial Prometheus/Grafana setup.

---

This project demonstrates strong skills in **NLP, model deployment, CI/CD, and MLOps practices**, showcasing readiness to handle production-grade ML systems.
