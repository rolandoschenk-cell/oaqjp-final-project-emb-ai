# Final Project

Flask application and Python package for the IBM/Coursera Emotion Detector
final project. The package calls the Watson NLP Emotion Predict API and returns
scores for anger, disgust, fear, joy, sadness, and the dominant emotion.

## Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 server.py
```

Open:

```text
http://localhost:5000/emotionDetector?textToAnalyze=I%20love%20this
```

## Tests

```bash
python3 -m unittest test_emotion_detection.py
```

## Pylint

```bash
pylint server.py
```
