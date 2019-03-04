import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertEmbeddings, BertModel, BertForSequenceClassification, CrossEntropyLoss, BertPooler
from pytorch_pretrained_bert.tokenization import BertTokenizer

class BertMaskForSequenceClassification(PreTrainedBertModel):
    def __init__(self, config, num_labels=2):
        """

        :param config:
        :param num_labels:
        :param max_offset:
        :param offset_emb: size of pos embedding, 0 to disable
        """
        print('model_mask')

        super(BertMaskForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def forward(self, input_ids, offset1, offset2, token_type_ids=None, attention_mask=None, labels=None):
        encode_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        mask1 = torch.zeros_like(encode_layer)
        mask1[offset1 == 10] = 1
        mask2 = torch.zeros_like(encode_layer)
        mask2[offset2 == 10] = 1
        maskcls = torch.zeros_like(encode_layer)
        maskcls[input_ids == self.tokenizer.vocab["[CLS]"]] = 1
        masksep = torch.zeros_like(encode_layer)
        masksep[input_ids == self.tokenizer.vocab["[SEP]"]] = 1
        mask = mask1 + mask2 + maskcls + masksep
        # lengths = torch.sum(mask, dim=1)
        # avg_layer = torch.sum(encode_layer * mask * attention_mask.unsqueeze(dim=2).float(), dim=1) / lengths
        # avg_layer = avg_layer.unsqueeze(dim=1)
        # pooled_output= self.pooler(avg_layer)
        max_layer, _ = torch.max(encode_layer * mask * attention_mask.unsqueeze(dim=2).float(), dim=1, keepdim=True)
        pooled_output= self.pooler(max_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits