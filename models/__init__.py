import sys
import os
for module_name in ["pytorch-pretrained-BERT", "LM-LSTM-CRF"]:
    path = os.path.join(os.path.dirname(__file__), module_name)
    sys.path.append(path)
    print("loading modules %s" % path)
