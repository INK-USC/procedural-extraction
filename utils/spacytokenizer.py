import spacy

"""
A cheaper tokenizer than corenlp
"""
_tokenizer = spacy.load('en')
_memory = dict()
def tokenize(sen): 
    if sen not in _memory:
        _memory[sen] = [str(tok) for tok in _tokenizer(sen)]
    return _memory[sen]