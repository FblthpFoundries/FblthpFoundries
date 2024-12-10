import pandas as pd
from tokenizers import Tokenizer
import re

def yoink(text, attribute):
        pattern = fr"<{attribute}>(.*?)<\\{attribute}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

data = pd.read_csv("corpus.csv")
tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")
data["tokens"] = data["card"].apply(lambda x: tokenizer.encode(x).ids)
data["mc"] = data["card"].apply(lambda x: yoink(x, "mc"))
data["power"] = data["card"].apply(lambda x: yoink(x, "power"))
data["toughness"] = data["card"].apply(lambda x: yoink(x, "toughness"))
data["cmc"] = data["mc"].apply(lambda x: sum([int(y) if y.isdigit() else 1 for y in x.replace(" ", "").replace("{", "").replace("}", "")]))

data["power"] = data["power"].apply(lambda x: int(x) if x.isdigit() else x)
data["toughness"] = data["toughness"].apply(lambda x: int(x) if x.isdigit() else x)

data.to_csv('labeled.csv', index=False)