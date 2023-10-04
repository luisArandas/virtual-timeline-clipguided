

# luis arandas 04-10-2023
# sequence terminal test

import os
import subprocess
import shutil

def clean_previous():
    # ! will erase
    folders_to_remove = ["models", "text_output", "tts_output"]
    for folder in folders_to_remove:
        folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Removed {folder} directory.")
        else:
            print(f"{folder} directory not found.")


def run_command(command):
    result = subprocess.run(command)    
    if result.returncode != 0:
        print(f"Error running {' '.join(command)}. Exiting tests.")
        exit(1)


def main():
    clean_previous()
    
    base_path = os.path.join(os.getcwd(), "text_output")
    tts_path = os.path.join(os.getcwd(), 'tts_output')
    length_tc = "00:02:00:00"

    commands_to_test = [
        ['python3', 'bio_1.py', '--text_dataset', 'shakespeare_short.txt'],
        ['python3', 'bio_2.py', '--summaries_dataset', os.path.join(base_path, 'sum_00001.txt')],
        ['python3', 'bio_3.py', '--speech_from_text', os.path.join(base_path, 'gen_00001.txt')],
        ['python3', 'bio_4.py', '--defined_length', '--audio_folder', tts_path, '--length', length_tc],
        ['python3', 'bio_5.py', '--timeline_file', os.path.join(base_path, 'timeline_00001.txt'), '--background_audio_file', os.path.join(os.getcwd(), 'background.wav')]
    ]

    for command in commands_to_test:
        print(f"Running {' '.join(command)}...")
        run_command(command)
        print(f"{' '.join(command)} completed successfully.\n")


if __name__ == "__main__":
    main()

