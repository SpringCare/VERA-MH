"""Unit tests for utils/utils.py"""

import pytest

from utils.utils import parse_key_value_list


@pytest.mark.unit
class TestParseKeyValueList:
    """Test suite for parse_key_value_list function"""

    def test_simple_string_values(self):
        """Test parsing simple string key-value pairs"""
        result = parse_key_value_list("name=John,city=Boston")
        assert result == {"name": "John", "city": "Boston"}

    def test_integer_values(self):
        """Test parsing integer values (should be converted from string)"""
        result = parse_key_value_list("age=25,count=100")
        assert result == {"age": 25, "count": 100}
        assert isinstance(result["age"], int)
        assert isinstance(result["count"], int)

    def test_float_values(self):
        """Test parsing float values"""
        result = parse_key_value_list("temperature=98.6,price=19.99")
        assert result == {"temperature": 98.6, "price": 19.99}
        assert isinstance(result["temperature"], float)
        assert isinstance(result["price"], float)

    def test_boolean_values(self):
        """Test parsing boolean values"""
        result = parse_key_value_list("active=True,enabled=False")
        assert result == {"active": True, "enabled": False}
        assert isinstance(result["active"], bool)
        assert isinstance(result["enabled"], bool)

    def test_none_value(self):
        """Test parsing None value"""
        result = parse_key_value_list("value=None")
        assert result == {"value": None}
        assert result["value"] is None

    def test_mixed_types(self):
        """Test parsing mixed types in single call"""
        result = parse_key_value_list(
            "name=Alice,age=30,height=5.5,active=True,notes=None"
        )
        assert result == {
            "name": "Alice",
            "age": 30,
            "height": 5.5,
            "active": True,
            "notes": None,
        }

    def test_empty_string_value(self):
        """Test parsing empty string value"""
        result = parse_key_value_list("key=")
        assert result == {"key": ""}
        assert result["key"] == ""

    def test_value_with_equals_sign(self):
        """Test parsing value that contains equals sign"""
        result = parse_key_value_list("equation=a=b+c")
        assert result == {"equation": "a=b+c"}

    def test_special_characters_in_string(self):
        """Test parsing strings with special characters"""
        result = parse_key_value_list("msg=hello world,path=/usr/local/bin")
        assert result == {"msg": "hello world", "path": "/usr/local/bin"}

    def test_list_value(self):
        """Test parsing list values without commas"""
        result = parse_key_value_list("items=[1]")
        assert result == {"items": [1]}
        assert isinstance(result["items"], list)

    def test_tuple_value(self):
        """Test parsing tuple values without commas"""
        result = parse_key_value_list("data=(42)")
        assert result == {"data": 42}
        # Note: (42) is parsed as int, not tuple

    def test_single_key_value_pair(self):
        """Test parsing single key-value pair"""
        result = parse_key_value_list("single=value")
        assert result == {"single": "value"}

    def test_negative_numbers(self):
        """Test parsing negative numbers"""
        result = parse_key_value_list("temp=-10,balance=-50.25")
        assert result == {"temp": -10, "balance": -50.25}
        assert isinstance(result["temp"], int)
        assert isinstance(result["balance"], float)

    def test_quoted_strings_remain_strings(self):
        """Test that quoted strings remain as strings"""
        result = parse_key_value_list("text='hello'")
        assert result == {"text": "hello"}
        assert isinstance(result["text"], str)

    def test_numeric_string_remains_string_when_quoted(self):
        """Test that quoted numeric values remain strings"""
        result = parse_key_value_list("zip='12345'")
        assert result == {"zip": "12345"}
        assert isinstance(result["zip"], str)


@pytest.mark.unit
class TestParseKeyValueListEdgeCases:
    """Test edge cases and error scenarios for parse_key_value_list"""

    def test_whitespace_in_values(self):
        """Test handling whitespace in values"""
        result = parse_key_value_list("name= John ,city= Boston ")
        # Whitespace is preserved in the values
        assert "name" in result
        assert "city" in result

    def test_invalid_format_missing_equals(self):
        """Test error handling when equals sign is missing"""
        with pytest.raises(ValueError):
            parse_key_value_list("invalid,format")

    def test_invalid_format_multiple_commas(self):
        """Test error handling with consecutive commas"""
        with pytest.raises(ValueError):
            parse_key_value_list("key=value,,another=value")

    def test_scientific_notation(self):
        """Test parsing scientific notation numbers"""
        result = parse_key_value_list("value=1e5")
        assert result == {"value": 100000.0}
        assert isinstance(result["value"], float)

    def test_keys_with_underscores(self):
        """Test keys with underscores and numbers"""
        result = parse_key_value_list("key_1=value1,key_2=value2")
        assert result == {"key_1": "value1", "key_2": "value2"}

    def test_url_value(self):
        """Test parsing URL as string value"""
        result = parse_key_value_list("url=https://example.com")
        assert result == {"url": "https://example.com"}
