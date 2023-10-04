

# luis arandas 27-10-2023
# $ python3 bio_2.py --summaries_dataset ./path/to/dataset

import os
import re
import argparse
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import torch
from transformers import GPTJForCausalLM, pipeline, GPT2Tokenizer


def setup_text_generation(device):
    _device = device

    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_length=100,
    ).to(_device)
    tokeniser = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    return model, tokeniser



def grammar_correction(text):
    if not text[-1] in [".", "!", "?"]:
        text += "."
    
    sentences = re.split('(?<=[.!?]) +', text)
    corrected_sentences = []
    for s in sentences:
        s_list = list(s)
        for i, char in enumerate(s_list):
            if char.isalpha():
                s_list[i] = char.upper()
                break
        corrected_sentence = ''.join(s_list)
        
        corrected_sentence = re.sub(r'(?<=[a-z])[A-Z]', lambda x: x.group(0).lower(), corrected_sentence)        
        corrected_sentences.append(corrected_sentence)
        
    corrected_text = ' '.join(corrected_sentences)
    return corrected_text




def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    separator_index = content.find("_____\n")
    if separator_index == -1:
        print("Separator '_____' not found in file.")
        return []
    text_content = content[separator_index + len("_____\n"):].strip()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text_content)
    return sentences




def export_generation_to_txt(text_data, dataset_path, model_name):
    if not os.path.exists(dataset_path):
        print(f"Invalid dataset path: {dataset_path}.")
        return
    
    if isinstance(text_data, str):
        text_data = [text_data]
    
    current_working_dir = os.getcwd()
    export_dir = os.path.join(current_working_dir, "text_output")

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_content = []
    for i, summary in enumerate(text_data):
        corrected_summary = grammar_correction(summary)

        structured_content = (
            f"time: {current_timestamp}\n"
            f"id: {i}\n"
            f"dataset path: {dataset_path}\n"
            f"model used: {model_name}\n"
            "_____\n\n"
            f"{corrected_summary}\n\n"
        )
        all_content.append(structured_content)
    
    existing_files = [f for f in os.listdir(export_dir) if f.startswith("gen_") and f.endswith(".txt")]
    if existing_files:
        max_number = max([int(f.split('_')[1].split('.')[0]) for f in existing_files])
        next_number = max_number + 1
    else:
        next_number = 1
    filename = f"gen_{next_number:05}.txt"
    filepath = os.path.join(export_dir, filename)
    with open(filepath, 'w') as file:
        file.writelines(all_content)

    print(f"Exported {len(text_data)} generated text to {filepath}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a custom dataset.')
    parser.add_argument('-d', '--summaries_dataset', type=str, required=True, help='Path to the generated dataset.')
    args = parser.parse_args()

    gpt_model, gpt_tokeniser = setup_text_generation(device="cuda")
    gpt_pipeline = pipeline("text-generation", model=gpt_model, tokenizer=gpt_tokeniser, device=0, pad_token_id=50256)
    
    new_text = []

    raw_data = read_sentences_from_file(args.summaries_dataset)
    
    if raw_data is not None:
        for i in raw_data:
            predicted_text = gpt_pipeline(i)
            lines = [line.strip() for line in predicted_text[0]['generated_text'].split('\n') if line.strip()]
            formatted_text = ' '.join(lines)
            new_text.append(formatted_text)
    
    merged = ''.join(new_text)
    export_generation_to_txt(merged, args.summaries_dataset, "EleutherAI/gpt-j-6B")

    print("Complete.")

