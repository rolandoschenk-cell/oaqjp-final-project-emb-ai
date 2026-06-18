"""Flask application for the Emotion Detector project."""

from flask import Flask, request

from EmotionDetection import emotion_detector


INVALID_TEXT_MESSAGE = "Invalid text! Please try again!"
PORT = 5001

app = Flask(__name__)


def format_emotion_response(result):
    """Format a successful emotion detection result for browser display."""
    dominant_emotion = result["dominant_emotion"]
    return (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {dominant_emotion}."
    )


@app.route("/emotionDetector")
def emotion_detector_route():
    """Analyze the textToAnalyze query parameter and return emotion scores."""
    text_to_analyze = request.args.get("textToAnalyze", "").strip()

    if not text_to_analyze:
        return INVALID_TEXT_MESSAGE

    result = emotion_detector(text_to_analyze)

    if result["dominant_emotion"] is None:
        return INVALID_TEXT_MESSAGE

    return format_emotion_response(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
