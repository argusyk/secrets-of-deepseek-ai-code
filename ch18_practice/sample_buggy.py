"""A deliberately flawed module used as input for the code-review assistant."""


def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket


def parse_config(text):
    try:
        return eval(text)
    except:
        return None


def find_user(users, name):
    for u in users:
        if u.get("name") == name:
            return u
    result = None
    if result == None:
        return {}
    return result
