import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertEmbeddings, BertModel, BertForSequenceClassification, CrossEntropyLoss

class BertOffsetEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position, token_type and offset embeddings.
    """
    def __init__(self, config, max_offset):
        super(BertOffsetEmbeddings, self).__init__(config)
        # offset from text_a and text_b
        self.offset1_embeddings = nn.Embedding(2*max_offset+1, config.hidden_size)
        self.offset2_embeddings = nn.Embedding(2*max_offset+1, config.hidden_size)

    def forward(self, input_ids, offset1, offset2, token_type_ids=None):
        # following BertEmbeddings, except adding offset embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        offset1_embeddings = self.offset1_embeddings(offset1)
        # offset2_embeddings = self.offset2_embeddings(offset2)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        #embeddings = embeddings + offset1_embeddings + offset2_embeddings
        embeddings = embeddings + offset1_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertOffsetembModel(BertModel):
    def __init__(self, config, max_offset):
        # replace embedding only
        super(BertOffsetembModel, self).__init__(config)
        self.embeddings = BertOffsetEmbeddings(config, max_offset)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, offset1=None, offset2=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, offset1, offset2, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertOffsetForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_labels=2, max_offset=10):
        super(BertOffsetForSequenceClassification, self).__init__(config, num_labels)
        # replace model only
        self.bert = BertOffsetembModel(config, max_offset)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, offset1, offset2, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, offset1=offset1, offset2=offset2)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
