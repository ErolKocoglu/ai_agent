import pytest
from unittest.mock import patch, MagicMock
from src.tools.joke_tool import get_joke

def test_get_joke_success():
    with patch('src.tools.joke_tool.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"joke": "Why did the chicken cross the road? To get to the other side."}
        mock_get.return_value = mock_response
        
        joke = get_joke("foo")
        assert joke == "Why did the chicken cross the road? To get to the other side."

def test_get_joke_failure():
    with patch('src.tools.joke_tool.requests.get') as mock_get:
        mock_get.side_effect = Exception("Network error")
        
        joke = get_joke("foo")
        assert "API error" in joke
