import unicodedata
import re


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,|])", r" \1", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.strip()

    w = '<start> ' + w + ' <end>'
    return w
