

# luis arandas 14-08-2023
# $ python3 bio_4.py --defined_length --length 00:02:00:00 --audio_folder tts_output/

import argparse

from timeline import VirtualTimeline, AudioProcessing


def convert_timecode_to_seconds(timecode, framerate=None):
    hours, minutes, seconds, frames = map(int, timecode.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if framerate:
        total_seconds += frames / framerate

    return total_seconds



def extract_lines_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    try:
        start_index = lines.index("_____\n") + 1
        extracted_lines = lines[start_index:]
    except ValueError:
        return []

    extracted_lines = [line.strip() for line in extracted_lines if line.strip()]
    return extracted_lines



def main():
    parser = argparse.ArgumentParser(description="Process the command.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--defined_length', action='store_true', help="User defined length.")
    group.add_argument('--derived_length', action='store_true', help="TTS derived length.")

    parser.add_argument('--length', type=str, help="Length in format HH:MM:SS (used with --defined_length).")
    parser.add_argument('--audio_folder', type=str, help="Filename following .txt pattern (captions or summaries).")
    parser.add_argument('--split_pad', type=str, help="Integer defining speech silence between sentences.")

    args = parser.parse_args()

    # instantiate the timeline object
    timeline = VirtualTimeline(fps=25)
    tts_output_full_path = str(args.audio_folder)

    AudioProcessing.process_folder(tts_output_full_path)
    audios_to_compute = AudioProcessing.loaded_audio_files

    timeline.compute_total_duration(audios_to_compute)

    txt_file_path = "./text_output/gen_00001.txt"
    prompt_list = [item for item in timeline.utils_extract_sentences_from_path(txt_file_path) if isinstance(item, str)]

    print("Timeline instantiated: ", timeline)
    print("Speech folder: ", tts_output_full_path)
    print("Audio files to compute: ", audios_to_compute)


    if args.defined_length:
        print("Computing method 1, audio sequence to last: ", args.length)        
        timeline.compute_sequencer_1(args.length, prompt_list)
        

    if args.derived_length:
        if args.split_pad == str(1):
            # sequence length is derived by how long it takes to say, + 1 sec
            print("Computing method 2, audio sequence to last according to padding: ", args.split_pad)
            timeline.compute_sequencer_2(prompt_list)
        
        if args.split_pad == str(5):
            # sequence length is derived by how long it takes to say, + 5 sec
            print("Computing method 2, audio sequence to last according to padding: ", args.split_pad)
            timeline.compute_sequencer_3(prompt_list)

    

if __name__ == "__main__":
    main()