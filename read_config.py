import json

def read_config(pth):
    with open(pth, "r") as f:
        result = dict(json.load(f))

    print(result)
    return result

if __name__ == "__main__":
    read_config("config.json")