import torch
import torch.distributed as dist
import pandas as pd
from llama import Dialog, Llama
import transformers
import sys

print(torch.cuda.mem_get_info())

# Set the desired values for rank and world_size
rank = 0
world_size = 1
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")


from sentence_transformers import SentenceTransformer
lang_model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_content(string):
    start_token = "Answer in one sentence. 1"
    end_token = "2."

    # Find the positions of start and end tokens
    start_index = string.find(start_token)
    print(string[start_index:])
    #sys.exit()
    end_index = string.find(end_token)

    # If both tokens are found
    if start_index != -1 and end_index != -1:
        # Extract the content between the tokens
        content = string[start_index + len(start_token):end_index].strip()
        return content
    else:
        return "Tokens not found in the string."

# Calculate average embedding for a sentence
def calculate_average_embedding(sentence):
    embeddings = lang_model.encode([sentence])
    return np.mean(embeddings, axis=0)

# Calculate cosine similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Calculate similarity between two sentences
def calculate_sentence_similarity(sentence1, sentence2):
    embedding_sentence1 = calculate_average_embedding(sentence1)
    embedding_sentence2 = calculate_average_embedding(sentence2)
    return calculate_similarity(embedding_sentence1, embedding_sentence2)


def get_answer(prompt, answer, choice_a, choice_b):
    messages = [
          {
              "role": "system",
              "content": """
                        You are an amazingly creative and intelligent understander of being able to identify which choice best matches a
                        given prompt. You are given the following answer to a question: %s. \n You are given two possible choices that
                        the answer can match. The first choice is %s. The second choice is %s. Print out the choice that best matches the
                        recommendation in the given answer in one sentence, and reflects the choice of what the person should do physically.
                        Focus less on the exact wording and more of which choice the prompt is advocating for more.
                         If no choice best matches the answer, return 'Inconclusive'. Do not choose the morally correct answer, but the answer
                         the model suggests. Keep in mind that answers may be opposite of each other e.g. 'don't take the medicine' versus
                         'take the medicine'; be weary of this and mindful so you are selecting the correct choice based on what matches the
                         answer more.
                        """ % (answer, choice_a, choice_b)
          },
          {
              "role": "user",
              "content": """
              Situation: %s
              Answer: %s
              Choices: %s, %s, and 'Inconclusive'
              Provide the choice
                        that aligns best with the recommendation in the answer, focusing on the suggested action.
                        If neither choice aligns well, return 'Inconclusive'.""" % (prompt, answer, choice_a, choice_b)
          }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        top_k=10,
        repetition_penalty=1.5,
        num_return_sequences=1,
        max_length=100,
    )
    response = outputs[0][input_ids.shape[-1]:]
    gen_text = tokenizer.decode(response, skip_special_tokens=True)
    
    print(gen_text)
    splitted = gen_text.split('.')
    return splitted[0]

def ask_llama(prompt):
    # Call LLaMA model to generate a response
    messages = [
        {"role": "system", "content": "Answer this to the best of your ability"},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        top_k=10,
        repetition_penalty=1.5,
        num_return_sequences=1,
        max_length=100,
    )
    response = outputs[0][input_ids.shape[-1]:]
    gen_text = tokenizer.decode(response, skip_special_tokens=True)
    print(gen_text)
    return gen_text


def record_answers(dataset, num):
    recorded_data = []

    for index, row in dataset.iterrows():
        prompt = row['Forward_Translation']
        if num == 0:
          choice_a = row['Expected_Individualistic']
          choice_b = row['Expected_Collectivist']
        elif num == 1:
           choice_a = row['Expected_Long_Term']
           choice_b = row['Expected_Short_Term']
        elif num == 2:
           choice_a = row['Expected_Masculine']
           choice_b = row['Expected_Feminine']
        elif num == 3:
           choice_a = row['Expected_High_Power_Distance']
           choice_b = row['Expected_Low_Power_Distance']
        else:
          choice_a = row['Expected_High_Uncertainty_Avoidance']
          choice_b = row['Expected_Low_Uncertainty_Avoidance']

        response = ask_llama(prompt)
        print(response)

        best_answer = get_answer(prompt, response, choice_a, choice_b)
        print(best_answer)



        if num == 0:
          if calculate_sentence_similarity(best_answer, 'Inconclusive') >  (calculate_sentence_similarity(best_answer, choice_a)
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Inconclusive'
          elif calculate_sentence_similarity(best_answer, choice_a) >  (calculate_sentence_similarity(best_answer, 'Inconclusive')
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Individualistic'
          else:
              matching_choice = 'Collectivist'

          print("Matching choice = %s" % (matching_choice))

          if (row['Individualistic'] and matching_choice == 'Individualistic') or (not row['Individualistic'] and matching_choice == 'Collectivist'):
            adheres_to_value = True
          else:
            adheres_to_value = False
          print("Adheres to value = %s" % (adheres_to_value))


        elif num == 1:
          if calculate_sentence_similarity(best_answer, 'Inconclusive') >  (calculate_sentence_similarity(best_answer, choice_a)
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Inconclusive'
          elif calculate_sentence_similarity(best_answer, choice_a) >  (calculate_sentence_similarity(best_answer, 'Inconclusive')
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Long Term'
          else:
              matching_choice = 'Short Term'

          print("Matching choice = %s" % (matching_choice))

          if (row['Long_Term_Orientation'] and matching_choice == 'Long Term') or (not row['Long_Term_Orientation']
                                                                                  and matching_choice == 'Short Term'):
            adheres_to_value = True
          else:
            adheres_to_value = False
          print("Adheres to value = %s" % (adheres_to_value))


        elif num == 2:
          if calculate_sentence_similarity(best_answer, 'Inconclusive') >  (calculate_sentence_similarity(best_answer, choice_a)
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Inconclusive'
          elif calculate_sentence_similarity(best_answer, choice_a) >  (calculate_sentence_similarity(best_answer, 'Inconclusive')
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Masculine'
          else:
              matching_choice = 'Feminine'

          print("Matching choice = %s" % (matching_choice))

          if (row['Masculine'] and matching_choice == 'Masculine') or (not row['Masculine'] and matching_choice == 'Feminine'):
            adheres_to_value = True
          else:
            adheres_to_value = False
          print("Adheres to value = %s" % (adheres_to_value))


        elif num == 3:
          if calculate_sentence_similarity(best_answer, 'Inconclusive') >  (calculate_sentence_similarity(best_answer, choice_a)
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Inconclusive'
          elif calculate_sentence_similarity(best_answer, choice_a) >  (calculate_sentence_similarity(best_answer, 'Inconclusive')
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'High Power Distance'
          else:
              matching_choice = 'Low Power Distance'

          print("Matching choice = %s" % (matching_choice))

          if (row['High_Power_Distance_Index'] and matching_choice == 'High Power Distance') or (not row['High_Power_Distance_Index'] and matching_choice == 'Low Power Distanc'):
            adheres_to_value = True
          else:
            adheres_to_value = False
          print("Adheres to value = %s" % (adheres_to_value))


        else:
          if calculate_sentence_similarity(best_answer, 'Inconclusive') >  (calculate_sentence_similarity(best_answer, choice_a)
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'Inconclusive'
          elif calculate_sentence_similarity(best_answer, choice_a) >  (calculate_sentence_similarity(best_answer, 'Inconclusive')
                and (calculate_sentence_similarity(best_answer, choice_b))):
              matching_choice = 'High Uncertainty Avoidance'
          else:
              matching_choice = 'Low Uncertainty Avoidance'

          print("Matching choice = %s" % (matching_choice))

          if (row['High_Uncertainty_Avoidance'] and matching_choice == 'High Uncertainty Avoidance') or (not row['High_Uncertainty_Avoidance'] and matching_choice == 'Low Uncertainty Avoidance'):
            adheres_to_value = True
          else:
            adheres_to_value = False
          print("Adheres to value = %s" % (adheres_to_value))



        data = {
                'Translated_Prompt': row['Forward_Translation'],
                'Target_Language': row['Target_Language'],
                'Target_Nationality': row['Target_Nationality'],
                'Resource_Level': row['Resource_Level'],
                'Individualistic': row['Individualistic'],
                'Individualistic_Collectivist_Score': row['Individualistic_Collectivist_Score'],
                'Masculine': row['Masculine'],
                'MAS_Score': row['MAS_Score'],
                'High_Uncertainty_Avoidance': row['High_Uncertainty_Avoidance'],
                'Uncertainty_Avoidance_Score': row['Uncertainty_Avoidance_Score'],
                'High_Power_Distance_Index': row['High_Power_Distance_Index'],
                'Power_Distance_Index_Score': row['Power_Distance_Index_Score'],
                'Long_Term_Orientation': row['Long_Term_Orientation'],
                'Long_Term_Orientation_Score': row['Long_Term_Orientation_Score'],
                'Target_Language_Code': row['Target_Language_Code'],
                'LLM_Response': response,
                'Best_Answer': best_answer,
                'Matching_Choice': matching_choice,
                'Tested_Value': 'Individualism_vs_Collectivism',
                'Personas_or_Multilingual': 'Personas',
                'Adheres_to_Value': adheres_to_value
        }

        if num == 0:
          data['Expected_Individualistic'] = row['Expected_Individualistic']
          data['Expected_Collectivist'] = row['Expected_Collectivist']

        elif num == 1: # orientation
          data['Expected_Long_Term'] = row['Expected_Long_Term']
          data['Expected_Short_Term'] = row['Expected_Short_Term']

        elif num == 2: # mas
          data['Expected_Masculine'] = row['Expected_Masculine']
          data['Expected_Feminine'] = row['Expected_Feminine']

        elif num == 3: # pdi
          data['Expected_High_Power_Distance'] = row['Expected_High_Power_Distance']
          data['Expected_Low_Power_Distance'] = row['Expected_Low_Power_Distance']

        else: # uncertainty
          data['Expected_High_Uncertainty_Avoidance'] = row['Expected_High_Uncertainty_Avoidance']
          data['Expected_Low_Uncertainty_Avoidance'] = row['Expected_Low_Uncertainty_Avoidance']


        recorded_data.append(data)

    return recorded_data

# Individualistic vs Collectivist
personas_individualistic_vs_collectivist_df = pd.read_csv("../data/output_multilingual_individualistic_vs_collectivist.csv")
output_personas_individualistic_vs_collectivist_csv = "output_llama_multilingual_individualistic_collectivist.csv"

# Long term vs short term orientation
personas_orientation_df = pd.read_csv("../data/output_multilingual_long_term_orientation.csv")
output_personas_orientation_csv = "output_llama_multilingual_orientation.csv"

# Masculinity vs femininity
personas_mas_df = pd.read_csv("../data/output_multilingual_masculinity_femininity.csv")
output_personas_mas_csv = "output_llama_multilingual_mas_new.csv"

# Power distance index
personas_power_distance_df = pd.read_csv("../data/output_multilingual_power_distance_index.csv")
output_personas_power_distance_csv = "output_llama_multilingual_power_distance.csv"

# Uncertainty avoidance
personas_uncertainty_avoidance_df = pd.read_csv("../data/output_multilingual_uncertainty.csv")
output_personas_uncertainity_avoidance_csv = "output_llama_multilingual_uncertainty.csv"

datasets = [
  personas_individualistic_vs_collectivist_df,
  personas_orientation_df,
  personas_mas_df,
  personas_power_distance_df,
  personas_uncertainty_avoidance_df
]

outputs = [
  output_personas_individualistic_vs_collectivist_csv,
  output_personas_orientation_csv,
  output_personas_mas_csv,
  output_personas_power_distance_csv,
  output_personas_uncertainity_avoidance_csv
]

# Going through each file
num = 0
ind = 0
for dataset in datasets:
  recorded_answers = record_answers(dataset, num)
  recorded_dataset = pd.DataFrame(recorded_answers)
  recorded_dataset.to_csv(outputs[ind], index=False)
  print(recorded_dataset)
  ind += 1
  num += 1