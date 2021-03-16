def truncate_string(string, length=100, add_dots=False):
    tokens = string.split()
    if len(tokens) > length:
        string = ' '.join(tokens[:length+1])
        if add_dots == True:
            string = ''.join([string,'...'])
    return string

class BERT_class:
    def __init__(self):
        # these are finetuned on SQuAD v2.0
        self.ALBERT_pretrained = ['ktrapeznikov/albert-xlarge-v2-squad-v2',
                 'ahotrod/albert_xxlargev1_squad2_512',
                 'elgeish/cs224n-squad2.0-albert-base-v2',
                 'twmkn9/albert-base-v2-squad2']
        # these are finetuned on SQuAD v1.1
        self.BERT_pretrained = ['bert-large-uncased-whole-word-masking-finetuned-squad',
                                'bert-large-cased-whole-word-masking-finetuned-squad']
        self.BERT_ranking = ['rsvp-ai/bertserini-bert-base-cmrc']
        self.EM_ALBERT = [84.41842836688285,
                   86.11134506864315,
                   78.94044093451794,
                   78.71010200723923]
        self.f1_ALBERT = [87.4628460501696,
                   89.35371214945009,
                   81.7724930324639,
                   81.89228117126069]
        self.EM_BERT = []
        self.f1_BERT = []
