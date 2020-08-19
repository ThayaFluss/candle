import os

def touch(path):
    if os.path.isfile(path):
        pass
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="UTF-8") as f:
            pass








