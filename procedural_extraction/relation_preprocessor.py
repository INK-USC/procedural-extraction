import logging
import pickle
import os
logger = logging.getLogger(__name__)

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class RelationProcessor(DataProcessor):
    """Processor for the Relation Context pkl."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.pkl")))
        with open(os.path.join(data_dir, "train.pkl"), 'rb') as f:
            obj = pickle.load(f)
        return self._create_examples(obj, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "dev.pkl"), 'rb') as f:
            obj = pickle.load(f)
        return self._create_examples(obj, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "test.pkl"), 'rb') as f:
            obj = pickle.load(f)
        return self._create_examples(obj, "test")

    def get_predict_examples(self, data_dir, id):
        with open(os.path.join(data_dir, "predict"+str(id)+".pkl"), 'rb') as f:
            obj = pickle.load(f)
        return self._create_examples(obj, "predict")

    def get_labels(self):
        """See base class."""
        return ["none", "next", "if"]

    def _create_examples(self, samples, set_type):
        """Creates examples for the training and dev sets."""
        for (i, sample) in enumerate(samples):
            guid = "%s-%s" % (set_type, i)
            sample.guid = guid
        return samples

def inflate_examples(examples, tokenizer, max_offset, ap_only=False):
    def convert_block(block):
        token, offset, alter_offset = list(), list(), list()
        for (idx, sen) in enumerate(block):
            if ap_only and idx != len(block) // 2:
                continue
            cur_token = tokenizer.tokenize(sen.text)
            token += cur_token
            offset += [truncate_offset(sen.offset)] * len(cur_token)
            alter_offset += [truncate_offset(sen.alter_offset)] * len(cur_token)
        return token, offset, alter_offset

    def truncate_offset(offset):
        return min(max_offset, max(-max_offset, offset)) + max_offset

    inflated_examples = list()
    for (ex_index, example) in enumerate(examples):
        token_a, offset1_a, offset2_a = convert_block(example.left)
        token_b, offset2_b, offset1_b = convert_block(example.right)
        inflated_examples.append(
            {
                'guid': example.guid, 
                'label': example.label,
                'tokens': (token_a, token_b), 
                'offset1': (offset1_a, offset1_b), 
                'offset2': (offset2_a, offset2_b)
            }
        )

    return inflated_examples