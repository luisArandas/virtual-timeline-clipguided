

""" 
    luis arandas 13-08-2023

    $ python3 bio_1.py --text_dataset ./path/to/dataset

    tests with (bio-env.txt)
""" 


import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from textsum.summarize import Summarizer

raw_data = []
raw_summaries = []
inference_parameters = None
summariser_model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"


def load_text_dataset(dataset_path):
    # loads the dataset from the provided path, file or folder
    data = []
    if os.path.isfile(dataset_path):
        with open(dataset_path, 'r') as file:
            data.append(file.read())
    elif os.path.isdir(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    data.append(file.read())
    else:
        print(f"The path {dataset_path} is neither a file nor a directory.")

    return data


def summarise_input_text_data(data, model_name):
    # runs every piece of text data through the booksum large model
    global raw_summaries
    summariser = Summarizer(model_name_or_path=model_name, token_batch_length=4096)
    global inference_parameters
    inference_parameters = summariser.get_inference_params()
    for i in data:
        print("Loaded text: ", i)
        summary = summariser.summarize_string(data)
        raw_summaries.append(summary)
        print(f'Generated summary: {summary}')



def export_summaries_to_txt(raw_summaries, dataset_path, model_name):
    # export each summary to independent txt file on same folder as dataset
    if not os.path.exists(dataset_path):
        print(f"Invalid dataset path: {dataset_path}.")
        return
    
    current_working_dir = os.getcwd()
    export_dir = os.path.join(current_working_dir, "text_output")

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for i, summary in enumerate(raw_summaries):
        filename = f"sum_{i+1:05}.txt"
        filepath = os.path.join(export_dir, filename)

        structured_content = (
            f"time: {current_timestamp}\n"
            f"id: {i}\n"
            f"dataset path: {dataset_path}\n"
            f"model used: {model_name}\n"
            "_____\n\n"
            f"{summary}"
        )

        with open(filepath, 'w') as file:
            file.write(structured_content)
    
    print(f"Exported {len(raw_summaries)} summaries to {export_dir}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a custom dataset.')
    parser.add_argument('-d', '--text_dataset', type=str, required=True, help='Path to the custom dataset.')
    args = parser.parse_args()

    raw_data = load_text_dataset(args.text_dataset)
    summarise_input_text_data(raw_data, summariser_model_name)
    export_summaries_to_txt(raw_summaries, dataset_path=args.text_dataset, model_name=summariser_model_name)
    
    print("Complete.")

    

