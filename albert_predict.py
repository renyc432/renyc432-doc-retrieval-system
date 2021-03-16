
import pandas as pd
import torch
from torch import argmax
from torch import topk
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadExample

from datetime import datetime

from util import truncate_string

class albert_predict:
    def __init__(self, model='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained(self.model, do_lower_case=True)
        self.albert = BertForQuestionAnswering.from_pretrained(self.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.albert.to(self.device)
        self.batch_size = 16
        self.topk = 5

# util
    def albert_choose_best_answers(self):
        answers_by_example = self.reranked.sort_values('albert score', ascending=False).drop_duplicates('example_index')
        answers_by_example.sort_values('example_index', ascending=True, inplace=True)
        self.reranked = answers_by_example

#    @profile
    def single_rerank(self, question, results):
        self.albert.eval()
        
        albert_scores = []
        answers = []
        for i, cont in enumerate(results['Hit']):
            tokens = self.tokenizer.encode_plus(text=question, 
                                                 text_pair=cont,
                                                 max_length=512,
                                                 add_special_tokens=True,
                                                 return_tensors='pt',
                                                 truncation=True,
                                                 return_attention_mask = False,
                                                 return_token_type_ids= False)
            with torch.no_grad():
                tokens.to(self.device)
                model_output = self.albert(**tokens)
            start_logits = model_output.start_logits
            end_logits = model_output.end_logits
            
            
# =============================================================================
# test whether this is faster than the code below
#             start = argmax(start_logits)
#             end = argmax(end_logits)
#             if start == end:
#                 start_top3 = topk(start_logits, k=3)
#                 end_top3 = topk(end_logits, k=3)
#                 # check if the second logits are the same
#                 if start_top3[1][0][1] == end_top3[1][0][1]:
#                     start = start_top3[1][0][2]
#                     end = end_top3[1][0][2]
#             
#             start_score_max = max(start_logits[0].detach().cpu().tolist())
#             end_score_max = max(end_logits[0].detach().cpu().tolist())
# =============================================================================
            
            start_topk = topk(start_logits, k=self.topk)
            end_topk = topk(end_logits, k=self.topk)
            
            for i in range(self.topk):
                start = start_topk[1][0][i]
                end = end_topk[1][0][i]
                if start != end:
                    break
            
            start_score_max = start_topk[0][0][0].cpu().tolist()
            end_score_max = end_topk[0][0][0].cpu().tolist()
            
            albert_score = start_score_max + end_score_max
                        
            input_ids = tokens['input_ids'].tolist()[0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            pred = tokens[start:end+1]
            answer = self.tokenizer.convert_tokens_to_string(pred)
            answer = answer.replace('[SEP]', '').replace('[CLS]','')
            if len(answer) == 0: 
                answer = 'No answer found'
            
            albert_scores.append(albert_score)
            answers.append(answer)
        
        self.reranked = pd.DataFrame({'answer': answers,
                                     'albert score': albert_scores})
        return self.reranked
            
    
    def batch_rerank(self, question, results):
        bert_input = []
        for i, cont in enumerate(results['Hit']):
            tokens = SquadExample(qas_id = i, 
                                  title = 'examples',
                                  question_text = question, 
                                  context_text = cont, 
                                  answer_text = None,
                                  answers=None,
                                  start_position_character = None,
                                  is_impossible=False)
            bert_input.append(tokens)
            
        print(f'\n length of bert_input: {len(bert_input)} \n')
        
        features, tensordataset = squad_convert_examples_to_features(examples = bert_input, 
                                                               tokenizer = self.tokenizer, 
                                                               max_seq_length = 512, 
                                                               doc_stride = 128, 
                                                               max_query_length = 64, 
                                                               is_training = False,
                                                               return_dataset='pt',
                                                               threads=1)
        
        print(f'\n length of features: {len(features)} \n', f'\n length of tensordataset: {len(tensordataset)} \n')
        bert_input_DL = DataLoader(tensordataset, batch_size=self.batch_size)
        print(f'number of batches in DL: {len(bert_input_DL)}')

        self.albert.eval()
        
        albert_scores = []
        answers = []
        example_indices = []
    
        for batch in bert_input_DL:    
            batch = tuple(b.to(self.device) for b in batch)
            
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                    }
                feature_indices = batch[3]
                output = self.albert(**inputs)
                
                start_logits = output.start_logits
                end_logits = output.end_logits
                
                for i in range(len(start_logits)):
                     start_score_max = max(start_logits[i].detach().cpu().tolist())
                     end_score_max = max(end_logits[i].detach().cpu().tolist())
                     albert_score = start_score_max + end_score_max
     
                     start = argmax(start_logits)
                     end = argmax(end_logits)
                     
                     feature = features[feature_indices[i]]
                     tokens = feature.tokens[start:end+1]
                     prediction = self.tokenizer.convert_tokens_to_string(tokens)
                     prediction = prediction.replace('[SEP]','').replace('[CLS]','')
                     if prediction == '':
                         prediction = 'No answer found'
                     
                     example_index = feature.example_index
                     
                     albert_scores.append(albert_score)
                     answers.append(prediction)
                     example_indices.append(example_index)
                            
        self.reranked = pd.DataFrame({'answer': answers,
                                     'albert score': albert_scores,
                                     'example_index': example_indices})
        self.albert_choose_best_answers()        

        
#    @profile
    def rerank(self, question, results):
        if (len(results) <= 100):
#            results['Hit'] = [truncate_string(cont) for cont in results['Hit']]
            return self.single_rerank(question, results)
        else:
            results['Hit'] = [truncate_string(cont) for cont in results['Hit']]
            self.batch_rerank(question, results)
            return self.reranked

        




