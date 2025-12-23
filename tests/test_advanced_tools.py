import pytest
from unittest.mock import patch, MagicMock
from src.tools.youtube_tool import search_youtube

def test_search_youtube_success():
    with patch('src.tools.youtube_tool.YoutubeSearch') as mock_search:
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = [
            {'title': 'Video 1', 'url_suffix': '/watch?v=123'},
            {'title': 'Video 2', 'url_suffix': '/watch?v=456'}
        ]
        mock_search.return_value = mock_instance
        
        result = search_youtube("test")
        assert "Video 1" in result
        assert "https://www.youtube.com/watch?v=123" in result

def test_search_youtube_no_results():
    with patch('src.tools.youtube_tool.YoutubeSearch') as mock_search:
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = []
        mock_search.return_value = mock_instance
        
        result = search_youtube("test")
        assert "No videos found" in result
