import spacy
from spacy.matcher import Matcher

spacy.require_gpu()

nlp = spacy.load("en_core_web_trf")
matcher = Matcher(nlp.vocab)
matcher.add("dog", [[{"LOWER": "dog"}]])

def replace_word(orig_text, replacement):
    tok = nlp(orig_text)
    text = ''
    buffer_start = 0
    for _, match_start, _ in matcher(tok):
        if match_start > buffer_start:  # If we've skipped over some tokens, let's add those in (with trailing whitespace if available)
            text += tok[buffer_start: match_start].text + tok[match_start - 1].whitespace_
        text += replacement + tok[match_start].whitespace_  # Replace token, with trailing whitespace if available
        buffer_start = match_start + 1
    text += tok[buffer_start:].text
    return text

print(replace_word("Hi this is my dog.", "Simba"))
print(replace_word("Hi this dog is my dog.", "Simba"))
