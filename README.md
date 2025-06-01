# Sentiment Analysis API (Dockerized with FastAPI + BERT)

A production-ready sentiment analysis API using a pre-trained BERT model from HuggingFace, powered by FastAPI and containerized with Docker.

## 🚀 Features
	•	Batch sentiment prediction (positive/negative) via REST API
	•	Powered by distilbert-base-uncased-finetuned-sst-2-english
	•	Lightweight FastAPI app for efficient inference
	•	Dockerized for easy deployment and portability

## 🧠 Model

This API uses:

distilbert-base-uncased-finetuned-sst-2-english

from HuggingFace’s Transformers library, fine-tuned on the SST-2 dataset for binary sentiment classification.

⸻

## Project Structure

<pre>
.
├── app/
│   ├── app.py             # FastAPI app definition
│   ├── model.py           # Model loading and prediction logic
│   └── requirements.txt   # Python dependencies
│
├── notebooks/             # Experimentation and model evaluation
├── tests/                 # Script to test API calls
├── Dockerfile             # Docker build instructions
├── .gitignore             # Git ignore rules
└── README.md              # Project overview
</pre>

⸻

## 🐋 Docker Usage

🔧 Build the Docker image

docker build -t sentiment-api .

## ▶️ Run the container

docker run -p 8000:8000 sentiment-api

Your API is now running at: http://127.0.0.1:8000

⸻

## 📫 API Usage

Endpoint: POST /predict

Request body:

{
  "texts": [
    "I love this movie!",
    "This is the worst thing ever."
  ]
}

Response:

[
  {
    "label": "positive",
    "confidence": 0.9998
  },
  {
    "label": "negative",
    "confidence": 0.9997
  }
]

## 🔬 Test with curl

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing performance!", "I hated the ending."]}'

## 🌐 Swagger UI

Interactive docs available at: http://127.0.0.1:8000/docs

⸻

🧪 Local Testing (Optional)

If running outside of Docker:

uvicorn app.app:app --reload


⸻

## ✅ Future Improvements
	•	Add support for fine-tuning on custom data
	•	Add GET /health endpoint
	•	Add frontend or deployment instructions

⸻