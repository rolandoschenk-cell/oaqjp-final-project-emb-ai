"""Client for the Watson NLP Emotion Predict API."""

import requests


API_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
REQUEST_TIMEOUT = (2, 5)
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")
NONE_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}


def emotion_detector(text_to_analyze):
    """Analyze text and return emotion scores plus the dominant emotion."""
    payload = {"raw_document": {"text": text_to_analyze}}
    headers = {"grpc-metadata-mm-model-id": MODEL_ID}

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 400:
            return dict(NONE_RESULT)

        response.raise_for_status()
    except requests.exceptions.RequestException:
        return dict(NONE_RESULT)

    response_data = response.json()
    emotions = response_data["emotionPredictions"][0]["emotion"]

    result = {emotion: float(emotions[emotion]) for emotion in EMOTIONS}
    result["dominant_emotion"] = max(result, key=result.get)
    return result
