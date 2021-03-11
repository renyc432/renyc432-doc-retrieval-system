
import pandas as pd

import torch
from torch import argmax
from torch.utils.data import DataLoader

from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadExample



results = pd.DataFrame({'Hit': ['COVID-19 research has overall low methodological quality thus far: case in point for chloroquine/hydroxychloroquine\nWhat is new? KEY FINDINGS: Clinical decision-makers must be informed by the best, most trustworthy, highest-quality, robust evidence. This translates into how much confidence we can have in the research findings and thus be optimally informed for decision-making. The estimates of effect in clinical research depends on the underlying research methodology. COVID-19 disease is presenting global health systems, clinicians, and patients grave challenges.',
                                    'No treatment or prophylaxis currently exists for COVID-19. The overall body of COVID-19 research is very flawed methodologically. An examination of hydroxychloroquine-azithromycin research findings due to the recent media focus revealed very low-quality methodology underpins the research. Vast amounts of time and resources are being allocated to COVID-19 research, and being potentially squandered. WHAT THIS ADDS TO WHAT WAS KNOWN: Flawed methodology and sub-optimal reporting of research findings could lead to biased estimates of effect. This could lead to treatment decisions that are not optimal based on biased estimates which could harm the patient.', 
                                    'This article provides specific suggestions for improving on the COVID-19 methods and reporting with a focus on the issues that researchers must consider in their methodology and reporting if we are to have confidence in the estimates of effect. Failure to consider harms in research could be detrimental to the patient. This article focuses on the potential harms when therapeutic agents such as hydroxychloroquine, are being considered. WHAT IS THE IMPLICATION AND WHAT SHOULD CHANGE NOW: Research thus far on finding an optimal therapeutic agent (s) for COVID-19 could be hampered by methodologically flawed research. COVID-19 researchers must immediately and acutely focus on improving their methodology and reporting.\nCOVID-19 disease is presenting global health systems, clinicians, and patients grave challenges.',
                                    'Coproducing Responses to COVID‐19 with Community‐Based Organizations: Lessons from Zhejiang Province, China\nZhejiang Province achieved one of the best records in containing the COVID‐19 pandemic in China, what lessons can the world learn from it? What roles do community‐based organizations play in its success story? Based on more than 100 interviews during and after the outbreak in Zhejiang, this article provides a roadmap of how community‐based organizations were involved in the three distinct stages of Zhejiang\'s responses to COVID‐19. We recommend that public sector leaders strategically leverage the strengths of community‐based organizations in multiple stages of COVID‐19 responses; incentivize volunteers to participate in epidemic prevention and control; provide data infrastructure and digital tracking platforms; and build trust and long‐term capacity of community‐based organizations. This article is protected by copyright. ',
                                    'COVID-19 convalescent plasma transfusion\n\n1.Promptness is essential since the strategy of administering the COVID-19 convalescent plasma has not yet been evaluated by randomized clinical trials. Data are only from a small number of case series with no control groups.2.Would patients improve after transfusion of the COVID-19 convalescent plasma despite receiving other antiviral and anti-inflammatory therapies?3.Would the use of the COVID-19 convalescent plasma transfusion reduce the infection-associated fatality rate and abbreviate the hospital stay?4.What would be the necessary dose of convalescent plasma to reach the clinical benefit? For how many days?5.What would be the adequate therapeutic titer of IgG and neutralization antibodies indicated to select the COVID-19 convalescent plasma donor?6.Is the plasma from donors with confirmed laboratory diagnosis of the COVID-19 and no clinical symptoms more protectible than those with clinical symptoms?7.Does the convalescent plasma from donors having a different virus genome infection have the protective effect for all patients with COVID-19?8.Besides neutralizing antibodies, what other factors could possibly be involved in inducing a clinical response?9.What is the best moment to transfuse the convalescent plasma? Should it be earlier (<10 days of symptoms) or is late (>10 days of beginning symptoms) transfusion of CP still effective?',
                                    'Donor Heart Selection During The COVID-19 Pandemic: A Case Study\nAs the offer was contemplated, a number of questions rapidly surfaced. These included (1) what is the likelihood of COVID-19 infection in a patient whose viral symptoms have resolved, and COVID-19 testing status is unknown, but a common respiratory virus has been identified; (2) if the donor were infected with COVID-19, what is the risk of donor-derived transmission involving a heart allograft (as opposed to lungs where the virus primarily resides); (3) is selective screening of donors acceptable, and if so, how, and who should be screened selectively, or should screening for COVID-19 in donors be performed universally in both symptomatic and asymptomatic donors, similar to other viruses; (4) would this particular donor family be willing to wait up to 3 additional days for the test result, and could the organ procurement organization keep the donor stable during this timeframe; (5) should lack of effective therapies for SARS-CoV-2 infection play into the decision to accept a heart with unknown COVID-19 status; (6) if the donor were infected, what is the risk to health care staff (e.g., procurement team, anesthesiologists, surgeons, operating room staff, and cardiac intensive care unit staff), and what is the risk to the recipient\'s family and relatives; and (7) given all these uncertainties, how do providers weigh the risk of COVID-19 infection against the competing risk of waitlist mortality.']})

results = pd.DataFrame({'Hit': ['COVID-19 research has overall low methodological quality thus far: case in point for chloroquine/hydroxychloroquine\nWhat is new? KEY FINDINGS: Clinical decision-makers must be informed by the best, most trustworthy, highest-quality, robust evidence. This translates into how much confidence we can have in the research findings and thus be optimally informed for decision-making. The estimates of effect in clinical research depends on the underlying research methodology. COVID-19 disease is presenting global health systems, clinicians, and patients grave challenges. No treatment or prophylaxis currently exists for COVID-19. The overall body of COVID-19 research is very flawed methodologically. An examination of hydroxychloroquine-azithromycin research findings due to the recent media focus revealed very low-quality methodology underpins the research. Vast amounts of time and resources are being allocated to COVID-19 research, and being potentially squandered. WHAT THIS ADDS TO WHAT WAS KNOWN: Flawed methodology and sub-optimal reporting of research findings could lead to biased estimates of effect. This could lead to treatment decisions that are not optimal based on biased estimates which could harm the patient. This article provides specific suggestions for improving on the COVID-19 methods and reporting with a focus on the issues that researchers must consider in their methodology and reporting if we are to have confidence in the estimates of effect. Failure to consider harms in research could be detrimental to the patient. This article focuses on the potential harms when therapeutic agents such as hydroxychloroquine, are being considered. WHAT IS THE IMPLICATION AND WHAT SHOULD CHANGE NOW: Research thus far on finding an optimal therapeutic agent (s) for COVID-19 could be hampered by methodologically flawed research. COVID-19 researchers must immediately and acutely focus on improving their methodology and reporting.\nThe urgency of the COVID-19 situation would also make it appropriate to be creative and move beyond the classical modalities and boundaries of academic research: what if a mobile app was made available by a respectable institution to allow randomizing any small number of consenting patients, collecting a small set of relevant covariates (age, sex, days since diagnosis, relevant comorbidities), by any doctor willing to participate in a chloroquine or hydroxychloroquine trial wherever the treatment is available for compassionate prescription (i.e. most of the world)? What if mortality in the two groups, masked as being treatment and control, was posted on a website every 100 patients reaching the outcome (recovered or dead) to transparently show if equipoise persists? What if the dataset at 1000 patients, or every 1000 patients if needed, was made publicly available for highly skilled statisticians to propose their interpretation? We would get 1000 patients every few days, and we would be receiving clinically sound results faster than any traditional study framework. Of course, this flexibility may not warrant publication in a top tier journal but could save thousands of lives. We need to use the most optimal methodology and not compromise on rigor but be willing to think outside of the box.',
                                'What we know so far about Coronavirus Disease 2019 in children: A meta‐analysis of 551 laboratory‐confirmed cases\nAIM: To summarize what we know so far about coronavirus disease (COVID‐19) in children. METHOD: We searched PubMed, Scientific Electronic Library Online, and Latin American and Caribbean Center on Health Sciences Information from 1 January 2020 to 4 May 2020. We selected randomized trials, observational studies, case series or case reports, and research letters of children ages birth to 18 years with laboratory‐confirmed COVID‐19. We conducted random‐effects meta‐analyses to calculate the weighted mean prevalence and 95% confidence interval (CI) or the weighted average means and 95% CI. RESULT: Forty‐six articles reporting 551 cases of COVID‐19 in children (aged 1 day‐17.5 years) were included. Eighty‐seven percent (95% CI: 77%‐95%) of patients had household exposure to COVID‐19. The most common symptoms and signs were fever (53%, 95% CI: 45%‐61%), cough (39%, 95% CI: 30%‐47%), and sore throat/pharyngeal erythema (14%, 95% CI: 4%‐28%); however, 18% (95% CI: 11%‐27%) of cases were asymptomatic. The most common radiographic and computed tomography (CT) findings were patchy consolidations (33%, 95% CI: 23%‐43%) and ground glass opacities (28%, 95% CI: 18%‐39%), but 36% (95% CI: 28%‐45%) of patients had normal CT images. Antiviral agents were given to 74% of patients (95% CI: 52%‐92%). Six patients, all with major underlying medical conditions, needed invasive mechanical ventilation, and one of them died. CONCLUSION: Previously healthy children with COVID‐19 have mild symptoms. The diagnosis is generally suspected from history of household exposure to COVID‐19 case. Children with COVID‐19 and major underlying condition are more likely to have severe/critical disease and poor prognosis, even death.\nThe results of this systematic review have implications for clinical practice and research. First, previously healthy children with COVID‐19 usually have mild symptoms and good prognosis. The diagnosis is generally suspected from history of household exposure to COVID‐19 case. For these patients, the management should focus on symptomatic and supportive care. In mild cases, unnecessary laboratory and imaging evaluation and unproven treatment should be avoided. Second, more attention should be given to children with COVID‐19 and major underlying medical conditions. These patients are more likely to have severe or critical disease and poor prognosis. Third, currently available evidence regarding COVID‐19 in children is mainly descriptive and anecdotal, and many questions remain unanswered. What are the risk factors for COVID‐19 in children? Why do children seem to be less affected by COVID‐19? What is the role of radiological imaging in the diagnosis and assessment of children with COVID‐19, and is there any advantage of CT scan over plain X‐ray? What are the effective treatments for children with COVID‐19? Could the WHO algorithm for the management of acute respiratory infections in children be applicable to patients with mild‐ to moderate COVID‐19, especially in low‐middle‐income countries? What are the prognosis factors for children with COVID‐19? What is the clinical implication of prolonged fecal shedding of SARS‐CoV‐2 RNA in children with COVID‐19? Further prospective multicenter studies are needed to answer these questions.',
                                'Answering the right questions for policymakers on COVID-19\nScientific models are crucial and useful for estimating impacts and prioritising response efforts across an entire state or region, but the tactical questions for each responder are often as well or better informed by data-driven, back-of-the-envelope estimates that are immediately relevant to the action that needs to be taken. These problems are just as acute in the COVID-19 response, if not more. The outpouring of basic descriptive epidemiology and national or global epidemic forecasting models has been key to pandemic response, but it has left many questions unanswered at smaller scales or in applied settings such as hospitals or town halls. We have identified the key questions that officials and experts in the USA need to be able to address and that can be addressed by currently available data or models (panel\n). These, we believe, are the questions that should most urgently be driving new analyses.PanelKey questions that officials and experts need to be able to address\n1. Clinical presentation and testing\nHow is the disease transmitted in different settings? How many cases are asymptomatic? How many cases are subclinical? How detectable is COVID-19 in syndromic surveillance data? What is the most effective use of diagnostic and serological testing, given low detection? How long does natural immunity last for those who have recovered? How does disease progression differ for different types of comorbidities? What explains differences in case fatality rate by country?\n2. Treatment: supplies, hospital beds, workforce\nHow many ventilators will each hospital need and when? Are the ventilators the limiting factor or is it the sedatives, beds, or the ability to staff those beds? Where in the hospital and for which tasks are different levels of personal protective equipment sufficient? What specific types of health-care specialties are most needed in regions with different types of comorbidities? What treatments are most successful for different types of patients and how can those be applied in practice?\n3. Non-pharmaceutical interventions: adherence and mobility\nWhat is the effectiveness of different types of non-pharmaceutical interventions and what makes them successful (eg, population density, percentage of people who comply, or degree to which they comply)? To what degree does spread appear to be driven by air travel versus other types of travel? What percentage of a community do we need to test to be able to shift back to contact tracing and to lift non-pharmaceutical interventions? What percentage of a hospital needs to be tested to shift back to isolation rooms and reduce personal protective equipment requirements?\n4. Public health response: ability to contact trace and identify exposures\nHow do we use the asymptomatic rate to inform when and how we deploy vaccines? At what level of herd immunity can we safely reopen schools? Can digital data accelerate contact tracing to a similar efficacy level to outbreaks that were contained early (eg, South Korea)? What legal or safety challenges do we need to address to be able to collect and use that data?\n5. Compound hazards and concurrent hazard planning\nHow do we structure emergency housing or evacuation for hurricanes or other natural disasters over the coming year without relying on mass care that might further spread COVID-19? How do we support homeless populations that are displaced? Do we evacuate hospitals with large numbers of contagious patients? How do we prioritise generators and fuel when every hospital is at capacity?']})
question = 'what is covid'



# this is necessary to compare bert_score of the same paragraph strided across batches
def albert_choose_best_answers(candidates):
    candidates = candidates.sort_values('albert score', ascending=False).drop_duplicates('example_index')
    candidates.sort_values('example_index', inplace=True)
    return candidates


def albert_rerank(question, results):
    # the following code predicts the start/end for each paragraph, but separately
    
    tokenizer = AlbertTokenizer.from_pretrained('ahotrod/albert_xxlargev1_squad2_512', do_lower_case=True)
    model_albert = AlbertForQuestionAnswering.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    features, tensordataset = squad_convert_examples_to_features(examples = bert_input, 
                                                           tokenizer = tokenizer, 
                                                           max_seq_length = 512, 
                                                           doc_stride = 128, 
                                                           max_query_length = 64, 
                                                           is_training = False,
                                                           return_dataset='pt')
    print(len(features), len(tensordataset))
    bert_input_DL = DataLoader(tensordataset, batch_size=4)
    
    model_albert.to(device)
    model_albert.eval()
    
    albert_scores = []
    answers = []
    example_indices = []

    for batch in bert_input_DL:    
        batch = tuple(b.to(device) for b in batch)
        
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
                }
            feature_indices = batch[3]
            output = model_albert(**inputs)
            
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
                 prediction = tokenizer.convert_tokens_to_string(tokens)
                 prediction = prediction.replace('[SEP]','').replace('[CLS]','')
                 if prediction == '':
                     prediction = 'No answer found'
                 
                 example_index = feature.example_index
                 
                 albert_scores.append(albert_score)
                 answers.append(prediction)
                 example_indices.append(example_index)
                        
    answers = pd.DataFrame({'answer': answers,
                            'albert score': albert_scores,
                            'example_index': example_indices})

    return albert_choose_best_answers(answers)


answers = albert_rerank(question,results)            
   
#[i.example_index for i in features]
         
    # =============================================================================
    # SquadExample(
    #             qas_id=tensor_dict["id"].numpy().decode("utf-8"),
    #             question_text=tensor_dict["question"].numpy().decode("utf-8"),
    #             context_text=tensor_dict["context"].numpy().decode("utf-8"),
    #             answer_text=answer,
    #             start_position_character=answer_start,
    #             title=tensor_dict["title"].numpy().decode("utf-8"),
    #             answers=answers,
    #         )
    # =============================================================================
    
    # =============================================================================
    # SquadFeatures(
    #     span["input_ids"],
    #     span["attention_mask"],
    #     span["token_type_ids"],
    #     cls_index,
    #     p_mask.tolist(),
    #     example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
    #     unique_id=0,
    #     paragraph_len=span["paragraph_len"],
    #     token_is_max_context=span["token_is_max_context"],
    #     tokens=span["tokens"],
    #     token_to_orig_map=span["token_to_orig_map"],
    #     start_position=start_position,
    #     end_position=end_position,
    #     is_impossible=span_is_impossible,
    #     qas_id=example.qas_id,
    # )
    # =============================================================================