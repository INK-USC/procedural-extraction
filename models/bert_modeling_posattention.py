import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertEmbeddings, BertModel, BertForSequenceClassification, CrossEntropyLoss

class BertPosattnForSequenceClassification(PreTrainedBertModel):
    def __init__(self, config, num_labels=2, max_offset=10, offset_emb=30):
        """

        :param config:
        :param num_labels:
        :param max_offset:
        :param offset_emb: size of pos embedding, 0 to disable
        """
        print('model_post attention')
        print('max_offset:', max_offset)
        print('offset_emb:', offset_emb)

        super(BertPosattnForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if offset_emb > 0:
            self.offset1_emb = nn.Embedding(2*max_offset+1, offset_emb)
            self.offset2_emb = nn.Embedding(2*max_offset+1, offset_emb)

        self.attn_layer_1 = nn.Linear((config.hidden_size + offset_emb) * 2, config.hidden_size)
        self.attn_tanh = nn.Tanh()
        self.attn_layer_2 = nn.Linear(config.hidden_size, 1)
        self.attn_softmax = nn.Softmax(dim=1)

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, offset1, offset2, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # Batch x Tok_pos x Hidden_dim
        batch_n, tok_n, hid_n = encoded_layers.size()
        # Batch x Hidden_dim ->  Batch x Tok_pos x Hidden_dim
        global_vec = pooled_output.unsqueeze(1).repeat(1, tok_n, 1)
        tensors_to_cat = [encoded_layers, global_vec]
        if hasattr(self, 'offset1_emb') and hasattr(self, 'offset2_emb'):
            tensors_to_cat += [self.offset1_emb(offset1), self.offset2_emb(offset2)]

        # (Tok_pos*Batch) x (Hidden_dim*2+offset_emb*2)
        attn_input = torch.cat(tensors_to_cat, 2).view(batch_n*tok_n, -1)
        # (Tok_pos*Batch) x Hidden_dim
        attn_1 = self.attn_layer_1(attn_input)
        attn_1 = self.attn_tanh(attn_1)
        # (Tok_pos*Batch) x 1 -> Batch x Tok
        attn_2 = self.attn_layer_2(attn_1).view(batch_n, tok_n)
        attn_weight = self.attn_softmax(attn_2)
        # Batch x Tok_pos x Hidden_dim -> Batch x Hiddem_dim
        weighted_layers = torch.sum(attn_weight.unsqueeze(2) * encoded_layers, dim=1, keepdim=True)
        pooled_output = self.bert.pooler(weighted_layers)
        pooled_output = self.dropout(pooled_output)
        # Batch x label_num
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits