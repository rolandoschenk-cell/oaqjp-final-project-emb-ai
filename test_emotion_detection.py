"""Unit tests for the emotion detection package."""

import unittest
from unittest.mock import Mock, patch

import requests
from EmotionDetection import emotion_detector
from server import INVALID_TEXT_MESSAGE, app


NONE_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}


def _mock_response(emotion_scores, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = [{"unused": "legacy"}]
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "emotionPredictions": [{"emotion": emotion_scores}]
    }
    return response


class TestEmotionDetector(unittest.TestCase):
    """Validate dominant emotion extraction from Watson-style responses."""

    def assert_dominant_emotion(self, scores, expected):
        """Assert that the detector returns the expected dominant emotion."""
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=_mock_response(scores),
        ):
            result = emotion_detector("Sample text")

        self.assertEqual(result["dominant_emotion"], expected)

    def test_joy(self):
        """Joy should be detected as the dominant emotion."""
        self.assert_dominant_emotion(
            {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.01,
                "joy": 0.95,
                "sadness": 0.02,
            },
            "joy",
        )

    def test_anger(self):
        """Anger should be detected as the dominant emotion."""
        self.assert_dominant_emotion(
            {
                "anger": 0.91,
                "disgust": 0.03,
                "fear": 0.02,
                "joy": 0.01,
                "sadness": 0.03,
            },
            "anger",
        )

    def test_disgust(self):
        """Disgust should be detected as the dominant emotion."""
        self.assert_dominant_emotion(
            {
                "anger": 0.04,
                "disgust": 0.89,
                "fear": 0.03,
                "joy": 0.01,
                "sadness": 0.03,
            },
            "disgust",
        )

    def test_sadness(self):
        """Sadness should be detected as the dominant emotion."""
        self.assert_dominant_emotion(
            {
                "anger": 0.03,
                "disgust": 0.02,
                "fear": 0.04,
                "joy": 0.01,
                "sadness": 0.90,
            },
            "sadness",
        )

    def test_fear(self):
        """Fear should be detected as the dominant emotion."""
        self.assert_dominant_emotion(
            {
                "anger": 0.03,
                "disgust": 0.02,
                "fear": 0.92,
                "joy": 0.01,
                "sadness": 0.02,
            },
            "fear",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_bad_request_returns_none_result(self, mock_post):
        """HTTP 400 responses should return the Coursera null-result shape."""
        response = Mock()
        response.status_code = 400
        mock_post.return_value = response

        result = emotion_detector("")

        self.assertEqual(result, NONE_RESULT)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_request_exception_returns_none_result(self, mock_post):
        """Network failures should return the Coursera null-result shape."""
        mock_post.side_effect = requests.exceptions.Timeout("API timeout")

        result = emotion_detector("The API is unavailable.")

        self.assertEqual(result, NONE_RESULT)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_flask_returns_invalid_text_when_api_unreachable(self, mock_post):
        """The Flask route should not fail with 500 when Watson is unreachable."""
        mock_post.side_effect = requests.exceptions.ConnectionError("No route")

        response = app.test_client().get(
            "/emotionDetector",
            query_string={"textToAnalyze": "Please analyze this."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), INVALID_TEXT_MESSAGE)


if __name__ == "__main__":
    unittest.main()
