import ast


def parse_key_value_list(arg):
    """Helper function to parse a list of key-value pairs into a dictionary."""
    d = {}
    for pair in arg.split(","):
        key, value = pair.split("=", 1)
        # Try Python literal parsing (handles ints, floats, booleans, None)
        # if it fails, we'll keep it as a stirng
        # https://docs.python.org/3/library/ast.html#ast.literal_eval
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Note: we are not logging the error here as we are leaving the value as a string
            pass
        d[key] = value
    return d
