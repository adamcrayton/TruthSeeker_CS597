from transformers import BertForSequenceClassification,RobertaForSequenceClassification, DistilBertForSequenceClassification, AlbertForSequenceClassification
def get_model(model_name):
    if model_name == 'roberta': 
        model = RobertaForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
            ignore_mismatched_sizes=True
        )
    elif model_name == 'distilbert':
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
            ignore_mismatched_sizes=True
        )
    elif model_name == 'albert':
        model = AlbertForSequenceClassification.from_pretrained(
            "textattack/albert-base-v2-imdb",
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
            ignore_mismatched_sizes=True
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    return model