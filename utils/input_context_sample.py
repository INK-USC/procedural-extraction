class InputContextSampleSentence(object):
    def __init__(self, text, offset, alter_offset):
        self.text = text
        self.offset = offset
        self.alter_offset = alter_offset


class InputContextSample(object):
    """A single training/test example with context for simple sequence classification."""

    def __init__(self, left_block, right_block, label=None, guid=None):
        self.left = left_block
        self.right = right_block
        self.label = label
        self.guid = guid

ISen = InputContextSampleSentence
IExample = InputContextSample