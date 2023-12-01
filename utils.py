from transformers import BertForSequenceClassification,RobertaForSequenceClassification, DistilBertForSequenceClassification, AlbertForSequenceClassification
def get_model(model_name, num_classes):
    if model_name == 'roberta': 
        model = RobertaForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion",
            num_labels = num_classes,
            output_attentions = False,
            output_hidden_states = False,
            ignore_mismatched_sizes=True
        )
    elif model_name == 'distilbert':
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels = num_classes,
            output_attentions = False,
            output_hidden_states = False,
            ignore_mismatched_sizes=True
        )
    elif model_name == 'albert':
        model = AlbertForSequenceClassification.from_pretrained(
            "textattack/albert-base-v2-imdb",
            num_labels = num_classes,
            output_attentions = False,
            output_hidden_states = False,
            ignore_mismatched_sizes=True
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_classes, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    return model

FOUR_WAY_MAPPING = {
    "True" : 0,
    "False": 1,
    "Mostly True": 2,
    "Mostly False": 3
}
TWO_WAY_MAPPING = {
    True: 0,
    False: 1
}

def get_integer_mapping_for_label(label_type):
    if label_type == '4-way-label':
        return FOUR_WAY_MAPPING
    elif label_type == '2-way-label':
        return TWO_WAY_MAPPING
