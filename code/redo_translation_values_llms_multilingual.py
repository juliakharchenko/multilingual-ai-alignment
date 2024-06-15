from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", token=True).to("cuda")

def translate_text(text, src_language, target_language):
    try:
        sentences = sent_tokenize(text)
        translated_sentences = []
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", token=True, src_lang=src_language)

        for sentence in sentences:
            try:
                inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
                generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_language])
                translated_sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                translated_sentences.append(translated_sentence)
            except Exception as sentence_error:
                print(f"Translation of sentence '{sentence}' failed: {sentence_error}")

        translated_text = ' '.join(translated_sentences)
        return translated_text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

def calculate_bleu(reference, candidate):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    smoother = SmoothingFunction()
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoother.method1)

def translate_csv(input_csv, output_csv, target_languages, num):
    df = input_csv
    rows = []

    for index, row in df.iterrows():
        for target_language in target_languages:
            forward_translation = translate_text(row['English_Prompt'], "eng_Latn", target_language)
            forward_bleu_score = calculate_bleu(row['English_Prompt'], forward_translation)
    
            back_translation = translate_text(forward_translation, target_language, "eng_Latn")
            back_bleu_score = calculate_bleu(row['English_Prompt'], back_translation)

            print(row['English_Prompt'])
            print("Translation into %s: %s" % (target_language, forward_translation))
            print('Back Translation into English: %s' % (back_translation))
            print('%s BLEU Score = %s' % (target_language, back_bleu_score))
                  
            if num == 0:
                rows.append([row['English_Prompt'], forward_translation, forward_bleu_score,
                             back_translation, back_bleu_score,
                             target_language, row['Expected_Individualistic'], row['Expected_Collectivist']])
            elif num == 1:
                rows.append([row['English_Prompt'], forward_translation, forward_bleu_score,
                             back_translation, back_bleu_score,
                             target_language, row['Expected_Long_Term'], row['Expected_Short_Term']])
            elif num == 2:
                rows.append([row['English_Prompt'], forward_translation, forward_bleu_score,
                             back_translation, back_bleu_score,
                             target_language, row['Expected_Masculine'], row['Expected_Feminine']])
            elif num == 3:
                rows.append([row['English_Prompt'], forward_translation, forward_bleu_score,
                             back_translation, back_bleu_score,
                             target_language, row['Expected_High_Power_Distance'], row['Expected_Low_Power_Distance']])
            else:
                rows.append([row['English_Prompt'], forward_translation, forward_bleu_score,
                             back_translation, back_bleu_score,
                             target_language, row['Expected_High_Uncertainty_Avoidance'], row['Expected_Low_Uncertainty_Avoidance']])

    if num == 0:
        translated_df = pd.DataFrame(rows, columns=['English_Prompt',
                      'Forward_Translation', 'Forward_BLEU_Score',
                      'Back_Translation', 'Back_BLEU_Score',
                      'Target_Language', 'Expected_Individualistic', 'Expected_Collectivist'])
    elif num == 1:
        translated_df = pd.DataFrame(rows, columns=['English_Prompt',
                      'Forward_Translation', 'Forward_BLEU_Score',
                      'Back_Translation', 'Back_BLEU_Score',
                      'Target_Language', 'Expected_Long_Term', 'Expected_Short_Term'])
    elif num == 2:
        translated_df = pd.DataFrame(rows, columns=['English_Prompt',
                      'Forward_Translation', 'Forward_BLEU_Score',
                      'Back_Translation', 'Back_BLEU_Score',
                      'Target_Language', 'Expected_Masculine', 'Expected_Feminine'])
    elif num == 3:
        translated_df = pd.DataFrame(rows, columns=['English_Prompt',
                      'Forward_Translation', 'Forward_BLEU_Score',
                      'Back_Translation', 'Back_BLEU_Score',
                      'Target_Language', 'Expected_High_Power_Distance', 'Expected_Low_Power_Distance'])
    else:
        translated_df = pd.DataFrame(rows, columns=['English_Prompt',
                      'Forward_Translation', 'Forward_BLEU_Score',
                      'Back_Translation', 'Back_BLEU_Score',
                      'Target_Language', 'Expected_High_Uncertainty_Avoidance', 'Expected_Low_Uncertainty_Avoidance'])
    return translated_df

# Individualistic vs Collectivist
multilingual_individualistic_vs_collectivist_df = pd.read_csv("../data/individualistic_vs_collectivist.csv")
output_multilingual_individualistic_vs_collectivist_csv = "output_multilingual_individualistic_collectivist.csv"

# Long term vs short term orientation
multilingual_orientation_df = pd.read_csv("../data/long_term_vs_short_term_orientation.csv")
output_multilingual_orientation_csv = "output_multilingual_orientation.csv"

# Masculinity vs femininity
multilingual_mas_df = pd.read_csv("../data/masculinity_femininity.csv")
output_multilingual_mas_csv = "output_multilingual_mas.csv"

# Power distance index
multilingual_power_distance_df = pd.read_csv("../data/power_distance_index.csv")
output_multilingual_power_distance_csv = "output_multilingual_power_distance.csv"

# Uncertainty avoidance
multilingual_uncertainty_avoidance_df = pd.read_csv("../data/uncertainty_avoidance.csv")
output_multilingual_uncertainity_avoidance_csv = "output_multilingual_uncertainty.csv"

datasets = [
  multilingual_individualistic_vs_collectivist_df,
  multilingual_orientation_df,
  multilingual_mas_df,
  multilingual_power_distance_df,
  multilingual_uncertainty_avoidance_df
]

outputs = [
  output_multilingual_individualistic_vs_collectivist_csv ,
  output_multilingual_orientation_csv,
  output_multilingual_mas_csv,
  output_multilingual_power_distance_csv,
  output_multilingual_uncertainity_avoidance_csv
]

target_languages = [
        "eng_Latn",  # English
        "deu_Latn",  # German
        "rus_Cyrl",  # Russian
        "jpn_Jpan",  # Japanese
        "fra_Latn",  # French
        "ita_Latn",  # Italian
        "zho_Hans",  # Chinese
        "ind_Latn",  # Indonesian
        "tur_Latn",  # Turkish
        "nld_Latn",  # Dutch
        "pol_Latn",  # Polish
        "pes_Arab",  # Persian
        "kor_Hang",  # Korean
        "ces_Latn",  # Czech
        "ukr_Cyrl",  # Ukrainian
        "hun_Latn",  # Hungarian
        "ell_Grek",  # Greek
        "ron_Latn",  # Romanian
        "swe_Latn",  # Swedish
        "heb_Hebr",  # Hebrew
        "dan_Latn",  # Danish
        "tha_Thai",  # Thai
        "fin_Latn",  # Finnish
        "bul_Cyrl",  # Bulgarian
        "kaz_Cyrl",  # Kazakh
        "hye_Armn",  # Armenian
        "kat_Geor",  # Georgian
        "als_Latn",  # Albanian
        "azj_Latn",  # Azerbaijani
        "zsm_Latn",  # Malay
        "khk_Cyrl",  # Mongolian
        "bel_Cyrl",  # Belarusian
        "hin_Deva",  # Hindi
        "afr_Latn",  # Afrikaans
        "isl_Latn",  # Icelandic
        "sin_Sinh"   # Sinhala
    ]

# Going through each file
num = 0
for _, (dataset, output) in enumerate(zip(datasets, outputs)):
    translated_df = translate_csv(dataset, output, target_languages, num)
    translated_df.to_csv(output, index=False)
    num += 1
