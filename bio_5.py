

# luis arandas 02-09-2023
# $ python3 bio_5.py --timeline_file /path/.txt

import os
import argparse

from timeline import VirtualTimeline, AudioProcessing


def main():

    parser = argparse.ArgumentParser(description="Process a timeline file.")

    parser.add_argument('--timeline_file', type=str, required=True, help='Path to the timeline file.')
    parser.add_argument('--background_audio_file', type=str, required=True, help='Path to the background audio file.')

    args = parser.parse_args()

    timeline_file_path = args.timeline_file

    timeline = VirtualTimeline(fps=25)

    print(f"Processing timeline file: {timeline_file_path}")
    timeline_dict = VirtualTimeline.utils_load_timeline_from_file(timeline_file_path)
    print("timeline dict loaded ", timeline_dict)

    length = list(timeline_dict.keys())[-1]
    AudioProcessing.trim_audio(args.background_audio_file, 'background_trimmed.wav', length/timeline._fps)
    AudioProcessing.convert_tts_to_16bit("tts_output/")

    # example usage
    
    tts_output_dir = "tts_output/"
    audio_files = [os.path.join(tts_output_dir, f) for f in os.listdir(tts_output_dir) if os.path.isfile(os.path.join(tts_output_dir, f)) and "_16bit" in f]

    if len(audio_files) < len([k for k, v in timeline_dict.items() if v != "Black."]):
        raise ValueError("Not enough audio files in 'tts_output/' for the non-'Black' keys in dict1.")

    inserts = {}
    file_index = 0
    for key, value in timeline_dict.items():
        if value != "Black.":
            inserts[key] = audio_files[file_index]
            file_index += 1

    print(inserts)    

    trimmed_background_path = "background_trimmed.wav"
    output_path = "output_mixed.wav"

    # here we should execute DDPM
    # guided_diffusion(), over length
    
    AudioProcessing.mix_audios_with_background(trimmed_background_path, inserts, output_path)

    print("Complete.")


if __name__ == "__main__":
    main()


