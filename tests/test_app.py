import unittest
import json
from flask_app.app import app  # Update this if your path is different

class FlaskAppJSONTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_generate_valid_input(self):
        """Test /generate route with valid input."""
        payload = {
            "user_input": "I feel amazing today!",
            "character": "<<MOIRA>>"
        }
        response = self.client.post(
            '/generate',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("character", data)
        self.assertIn("detected_emotion", data)
        self.assertIn("response", data)

    def test_generate_missing_input(self):
        """Test /generate route with missing 'user_input'."""
        payload = {
            "character": "<<DAVID>>"
        }
        response = self.client.post(
            '/generate',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"], "Missing 'user_input'")

if __name__ == '__main__':
    unittest.main()
