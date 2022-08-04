import unicodedata

def greek_to_ascii(symbol):
    greek, size, letter, what, *with_tonos = unicodedata.name(symbol).split()
    assert greek, letter == ("GREEK", "LETTER")
    return what.lower() if size == "SMALL" else what.title()

def format_string(string):
    new_string = []
    for a in string:
        try:
            new_string.append(greek_to_ascii(a))
        except:
            new_string.append(a)
    return "".join(new_string)
