

""" 
    luis arandas 14-07-2023

    $ python3 bio_3.py --speech_from_text ./path/to/dataset.txt

    tests with (kdiff)
""" 


import os
import argparse

import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

# import nltk
# nltk.download('punkt')
from nltk import tokenize

from timeline import VirtualTimeline


def setup_audio_output():
    current_working_dir = os.getcwd()
    audio_output_path = os.path.join(current_working_dir, "tts_output")
    if not os.path.exists(audio_output_path):
        os.makedirs(audio_output_path)
    return audio_output_path


def text_to_speech(loaded_summaries):
    audio_output_dir = setup_audio_output()

    # using: tacotron2, hifi-gan and speechbrain
    
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="models/tmpdir_tts")
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="models/tmpdir_vocoder")

    folder_count = 0

    for i in loaded_summaries:
        x = tokenize.sent_tokenize(i)
        for j in x:
            mel_output, mel_length, alignment = tacotron2.encode_text(j)
            waveforms = hifi_gan.decode_batch(mel_output)
            new_name = os.path.join(audio_output_dir, f'gen_{folder_count}_TTS.wav')
            torchaudio.save(new_name, waveforms.squeeze(1), 22050)
            folder_count += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech_from_text', type=str, help='input text for TTS, direct', default='./text_output/gen_00001.txt')
    args = parser.parse_args()

    timeline = VirtualTimeline(fps=25)
    loaded_summaries = timeline.utils_extract_sentences_from_path(args.speech_from_text)
    print("Generating TTS for loaded summaries: ", loaded_summaries)
    text_to_speech(loaded_summaries=loaded_summaries)
    
    print("Complete.")


