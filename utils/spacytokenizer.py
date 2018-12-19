import logging

import spacy

log = logging.getLogger(__name__)

class Tokenizer(object):
    """
    A cheaper tokenizer than corenlp
    """
    def __init__(self):
        log.info("loading spacy tokenize model")
        self._tokenizer = spacy.load('en')
        self._memory = dict()

    def tokenize(self, sen):
        """
        Tokenize sentence using spacy
        """
        if sen not in self._memory:
            self._memory[sen] = [str(tok) for tok in self._tokenizer(sen)]
        return self._memory[sen]