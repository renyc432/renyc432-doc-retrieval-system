
import pandas as pd
import numpy as np

import torch
from torch import argmax
from torch.utils.data import DataLoader

from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadExample

from datetime import datetime

class albert_predict:
    def __init__(self):
        self.QA_pretrained = ['ktrapeznikov/albert-xlarge-v2-squad-v2',
                 'ahotrod/albert_xxlargev1_squad2_512']
        self.EM = [84.41842836688285,
                   86.11134506864315]
        self.f1 = [87.4628460501696,
                   89.35371214945009]

        self.tokenizer = AlbertTokenizer.from_pretrained(self.QA_pretrained[1], do_lower_case=True)
        self.albert = AlbertForQuestionAnswering.from_pretrained(self.QA_pretrained[1])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = 16

    def predict(self, question, context):
        self.model_albert.to(self.device)
        self.modle_albert.eval()
        tokens = self.tokenizer_albert.encode_plus(text=question, 
                                             text_pair=context,
                                             max_length=512,
                                             add_special_tokens=True,
                                             return_tensors='pt',
#                                             truncation=True,
                                             return_attention_mask = False,
                                             return_token_type_ids= False)
        
        model_output = self.model_albert(**tokens)
        start_logits = model_output.start_logits
        end_logits = model_output.end_logits
        
        start = argmax(start_logits)
        end = argmax(end_logits)
        
        input_ids = tokens['input_ids'].tolist()[0]
        tokens = self.tokenizer_albert.convert_ids_to_tokens(input_ids)
        pred = tokens[start:end+1]
        prediction = ' '.join(pred).replace('▁','').strip()
        prediction = prediction.replace('[SEP]','').replace('[CLS]','')
        
        if len(prediction) == 0: 
            return 'Could not find a satisfactory answer in this article. However, that is not to say there is not one. Check it out for yourself.'
        return prediction
    
    def albert_choose_best_answers(self):
        answers_by_example = self.reranked.sort_values('albert score', ascending=False).drop_duplicates('example_index')
        answers_by_example.sort_values('example_index', ascending=True, inplace=True)
        self.reranked = answers_by_example
        
    # use albert-squad to rerank; maybe also highlight?
    # result: 
    def rerank(self, question, results):
       
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
                                                               max_seq_length = 512-128, 
                                                               doc_stride = 128, 
                                                               max_query_length = 64, 
                                                               is_training = False,
                                                               return_dataset='pt')
        
        print(f'\n length of features: {len(features)} \n', f'\n length of tensordataset: {len(tensordataset)} \n')
        
        bert_input_DL = DataLoader(tensordataset, batch_size=self.batch_size)
        
        print(f'number of batches in DL: {len(bert_input_DL)}')
        

        self.albert.to(self.device)
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
        
        return self.reranked
            
        

    
# =============================================================================
#         bert_input = []
#         for ques, para in zip([self.question]*len(bert_input),result['Hit']):
#             tokens = self.tokenizer_albert.encode_plus(text=ques,
#                                                        text_pair=para,
#                                                        max_length=512,
#                                                        add_special_tokens=True,
#                                                        return_tensors='pt',
#                                                        truncation=True)
#             bert_input.append(tokens)
#             
#         bert_input_DL = DataLoader(bert_input, batch_size=self.batch_size)
#         
#         start_logits_max = []
#         end_logits_max = []
#         answers = []
#         for batch in bert_input_DL:
#             batch = (b.to(self.device) for b in batch)
#             with torch.no_grad():
#                 output = self.model_albert(**batch)
#                 start_logits = output.start_logits
#                 end_logits = output.end_logits
#                 
#                 start = argmax(start_logits)
#                 end = argmax(end_logits)
#                 
#                 input_ids = tokens['input_ids'].tolist()[0]
#                 tokens = self.tokenizer_albert.convert_ids_to_tokens(input_ids)
#                 pred = tokens[start:end+1]
#                 prediction = ' '.join(pred).replace('▁','').strip()
#                 prediction = prediction.replace('[SEP]','').replace('[CLS]','')
#                 answers.append(prediction)
#         
#         return start_logits_max, end_logits_max, answers
# =============================================================================





