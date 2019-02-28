import re
import os.path

import numpy as np

import utils

class TargetProcessor(object):
    """
    Extract structured infos from natural text
    Init class for one document
    Invoke process for each line
    """
    def __init__(self):
        """
        Init for doc
        """
        self.patt_step = re.compile(r'^\s*(Task|Step)\s+([^:]+):\s+')
        self.patt_branch = re.compile(r'^\s*(BRANCH)\s+([^:]+):\s+')
        self.patt_goto = re.compile(r'(GOTO)\s+(Task|Step|BRANCH)\s+([^\s]+)')

        self.cur_norm_pos = ['S']
        self.cur_norm_type = None

    def text(self, intext):
        """
        Init for each splited sentence
        """
        self.ori_text = intext
        self.line_text = intext
        self.cur_pos = self.cur_norm_pos
        self.cur_type = self.cur_norm_type
        self.next_type = None
        self.next_pos = None
        return self

    def step(self):
        """
        Extract task & step infos
        "Task A.3.1: blablabla" -> ['A', 3, 1]
        """
        text = self.line_text
        patt = self.patt_step
        m = patt.search(text)
        if m is None:
            # If nothing found, follow previous position
            pass
        else:
            # else, use current position
            self.cur_type = m.group(1)
            self.cur_pos = utils.convert_int(m.group(2).split('.'))
            self.cur_norm_type = self.cur_type
            self.cur_norm_pos = self.cur_pos
            self.line_text = patt.sub('', text)
        return self

    def branch(self):
        """
        Extract Branch infos
        "BRANCH A: blablabla" -> ['A', 3, 1]
        """
        text = self.line_text
        patt = self.patt_branch
        m = patt.search(text)
        if m is None:
            # If nothing found, follow previous position
            pass
        else:
            # else, use current position
            self.cur_type = m.group(1)
            self.cur_pos = m.group(2)
            self.cur_norm_type = self.cur_type
            self.cur_norm_pos = self.cur_pos
            self.line_text = patt.sub('', text)
        return self

    def goto(self):
        """
        Extract goto infos
        "GOTO BRANCH A"
        """
        text = self.line_text
        patt = self.patt_goto
        m = patt.search(text)
        if m is None:
            # If nothing found, do nothing
            pass
        else:
            # else, update result
            self.cur_type = m.group(1)
            self.next_type = m.group(2)
            self.next_pos = m.group(3)
            self.line_text = patt.sub('', text)
        return self

    def getResult(self):
        return {
            'ori_text': self.ori_text,
            'text': self.line_text,
            'cur_pos': self.cur_pos,
            'cur_type': self.cur_type,
            'next_type': self.next_type,
            'next_pos': self.next_pos
        }

    def process(self, text):
        """
        Process a sub sentence
        """
        res = self.text(text).step().branch().goto().getResult()
        return res