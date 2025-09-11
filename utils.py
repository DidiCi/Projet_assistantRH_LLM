import json

def safe_parse(content):
    try:
        return json.loads(content)
    except Exception:
        return {}