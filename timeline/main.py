

# luis arandas 26-02-2023
# simple matrix timeline object

import math
import re
import time

import torch
import transformers

from essential_generators import DocumentGenerator # I will replace this for embedding randomization
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from speechbrain.pretrained import SpectralMaskEnhancement

from cleantext import clean
from tqdm.auto import tqdm

# I NEED TO ADD FORMAL RATIOS TO THIS


class VirtualTimelineStore(object):

    """ 
        main timeline object
        defines internal variables for each sequence render
        these should schedule events and render-related setup
    """

    def __init__(self):
        
        self.batch_folder = ""
        self.data = [[[],[]],{}] # simple matrix
        # self.data[0] = [[[],[]], {}]
        self.film_length_secs = 0
        self.fps = 25
        self.audio_sample_rate = 22050 # quick on macos 12.2

        self.max_frames = 0
        self.starting_text = ""

        self.short_summary = "" # led(t)
        self.short_summary_sentences = []
        self.short_summary_sentences_audio_frames = []
        self.short_summary_sentences_audio_lengths = []
        self.short_summary_sentences_audio_files_nr = 0
        self.short_summary_sentences_audio_sec_total = 0
        self.short_summary_sentences_audio_sec_single = []
        self.short_summary_sentences_diffusion_frames = 0

        self.keywords = [] # keyword_extract(t)
        self.keywords_audio_frames = []
        self.keywords_audio_lengths = []
        self.keywords_nr_audio_files = 0

        # language models 

        self.grammar_detector = transformers.pipeline("text-classification", "textattack/roberta-base-CoLA")
        self.grammar_corrector = transformers.pipeline("text2text-generation", model="pszemraj/flan-t5-large-grammar-synthesis")

        self.summariser_model, self.summariser_tokenizer = self.load_model_and_tokenizer("pszemraj/led-large-book-summary")
        self.summariser_model_sm, self.summariser_tokenizer_sm = self.load_model_and_tokenizer("pszemraj/led-base-book-summary")

        # audio models

        self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="models/tmpdir_tts")
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="models/tmpdir_vocoder")
        self.sm_enh = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank", savedir="models/tmpdir_enhancer")

        # self.long_prompts = []
        # self.nr_audio_frames = []
        # self.total_audio_files = 0
        # self.audio_lengths_secs = []
        # self.total_video_frames = 0

        self.tokens_to_exclude = "religion, love hearts, crosses, eyes, jesus" 
        # the loss should computed against values of this string later
        # should work on all approximation iterations

    """
        three different sequencers of matrix data[1] -> (dict)
        should be straightforward runs reading from the object
    """

    def sequencer_1(self):

        # calculate the max frames based on user input and divide
        # the raw led(t) in sentences across the length, automatic
        
        self.max_frames = int(self.film_length_secs * self.fps)
        nr_frames_w_prompt = int(self.max_frames / len(self.short_summary_sentences))
        equal_distances = list(range(0, int(self.max_frames), int(nr_frames_w_prompt)))
        self.update_dict(self.data[1], equal_distances, self.short_summary_sentences)
        self.data[1] = {k: v for k, v in self.data[1].items() if v != ""}

    def sequencer_2(self):

        # calculate the max frames based on generated tts(led(t)) to have a speech sequence
        # good to experiment with variable skip step percentages, the max_frames will add 1 second each side
        
        self.short_summary_sentences_audio_files_nr = len(self.short_summary_sentences_audio_frames)
        self.short_summary_sentences_audio_sec_single = [int(math.ceil(x / self.audio_sample_rate)) for x in self.short_summary_sentences_audio_frames]
        total_generated_audio_frames = sum(self.short_summary_sentences_audio_frames)

        self.short_summary_sentences_audio_sec_total = int(math.ceil(total_generated_audio_frames / self.audio_sample_rate))
        next_multiple_of_ten = math.ceil(self.short_summary_sentences_audio_sec_total / 10) * 10
        # this doesn't map in the specific spoken words with the diffused keyframes, but to the whole buffer
        self.short_summary_sentences_diffusion_frames = int(math.ceil(next_multiple_of_ten * self.fps))
        self.max_frames = self.short_summary_sentences_diffusion_frames + 2 * self.fps # add two blocks of second
        # now dividing equally the prompts over the period in which speech exists
        # if total seconds of tts() is 10, we will diffuse 12 seconds one each side
        _prompt_keys = [0, self.fps, self.max_frames-self.fps, self.max_frames]
        # calculate incremental distance for new elements of _prompt_keys[1] and [2]
        incremental_distance = int((_prompt_keys[2] - _prompt_keys[1]) / ((self.short_summary_sentences_audio_files_nr-1) + 1))
        new_elements = [_prompt_keys[1] + incremental_distance * (i+1) for i in range(self.short_summary_sentences_audio_files_nr-1)]
        if len(new_elements) > 0:
            _prompt_keys[1:-1] = new_elements # check if they distribute equally
            self.update_dict(self.data[1], _prompt_keys, self.short_summary_sentences)
            self.data[1] = {k: v for k, v in self.data[1].items() if v != ""}
        else:
            self.update_dict(self.data[1], _prompt_keys, self.short_summary_sentences)
            self.data[1] = {k: v for k, v in self.data[1].items() if v != ""}

    def sequencer_3(self):

        # based on user length definition feed single words to the diffusion system from keyword_extract()
        # compute 5 seconds for each word plus two blocks of second at each side
        
        self.max_frames = int(self.film_length_secs * self.fps)
        pick_first_three = [tup[0] for tup in self.keywords[:3]] # also removing scores from tuple
        _prompt_keys = [0, self.fps, self.max_frames-self.fps, self.max_frames]
        incremental_distance = int((_prompt_keys[2] - _prompt_keys[1]) / (len(pick_first_three) + 1))
        new_elements = [_prompt_keys[1] + incremental_distance * (i+1) for i in range(len(pick_first_three))]
        if len(new_elements) > 0:
            x_idx, y_idx = _prompt_keys[1], _prompt_keys[2]
            part_1, part_2 = _prompt_keys[:x_idx], _prompt_keys[y_idx:]
            _prompt_keys = part_1 + new_elements + part_2
            self.update_dict(self.data[1], _prompt_keys, pick_first_three)
            self.data[1] = {k: v for k, v in self.data[1].items() if v != ""}
        else:
            self.update_dict(self.data[1], _prompt_keys, pick_first_three)
            self.data[1] = {k: v for k, v in self.data[1].items() if v != ""}

    def sequencer_random(self):

        # just create a random sequence of text prompts
        # foreach second defined by user input

        self.data[1].clear()
        if self.film_length_secs > 0:
            x =  DocumentGenerator()
            key_frame_arr = [i*25 for i in range(self.max_frames//25)] # + [timeline.max_frames]
            print("key frame arr ", key_frame_arr)
            for key in key_frame_arr:
                # Generate a random string of length 10
                random_data = x.sentence()
                self.data[1][key] = random_data

    """
        establish matrix[0] and further tts() for soundtrack entire length
    """

    def tts_prompts_by_keyword_extract(self): # passing reference
        for idx, item in enumerate(self.keywords):
            if len(item) != 0:
                mel_output, mel_length, alignment = self.tacotron2.encode_text(str(item)) 
                waveform = self.hifi_gan.decode_batch(mel_output)
                self.keywords_audio_frames.append(waveform.shape[2])
                self.keywords_audio_lengths.append(waveform.shape[2] / self.audio_sample_rate) 
                # process
                filename = f'{self.batch_folder}/keyword_sentence_{idx}_TTS.wav'
                torchaudio.save(filename, waveform.squeeze(1), self.audio_sample_rate)

    def tts_prompts_by_sentence(self):
        for idx, item in enumerate(self.short_summary_sentences):
            if len(item) != 0:
                if not str(item).isnumeric(): # got some issues on dates
                    # timeline.long_prompts.append(item) # define the image diffusion prompts
                    mel_output, mel_length, alignment = self.tacotron2.encode_text(str(item))
                    waveform = self.hifi_gan.decode_batch(mel_output)
                    self.short_summary_sentences_audio_frames.append(waveform.shape[2])
                    self.short_summary_sentences_audio_lengths.append(waveform.shape[2] / self.audio_sample_rate)
                    # process
                    filename = f'{self.batch_folder}/short_sum_sentence_{idx}_TTS.wav'
                    torchaudio.save(filename, waveform.squeeze(1), self.audio_sample_rate)
    
    """
        NLP related functions and methods
        summariser is forked from huggingface -> pszemraj/summarize-long-text
    """

    def load_model_and_tokenizer(self, model_name):
        # returns: AutoModelForSeq2SeqLM: the model ; AutoTokenizer: the tokenizer
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True, use_cache=False)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = model.to("cuda") if torch.cuda.is_available() else model
        return model, tokenizer

    def summarise_and_score(self, ids, mask, model, tokenizer, **kwargs):
        # given a batch of ids and a mask, return a summary and a score for the summary
        # model   (): the model to use for summarization
        # tokenizer (): the tokenizer to use for summarization
        
        ids = ids[None, :] # ids (): the batch of ids
        mask = mask[None, :] # mask (): the attention mask for the batch
        input_ids = ids.to("cuda") if torch.cuda.is_available() else ids
        attention_mask = mask.to("cuda") if torch.cuda.is_available() else mask
        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        summary_pred_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, output_scores=True, return_dict_in_generate=True, **kwargs)
        summary = tokenizer.batch_decode(summary_pred_ids.sequences, skip_special_tokens=True, remove_invalid_values=True)
        score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)
        return summary, score
    
    def summarise_via_tokenbatches(self, input_text: str, model, tokenizer, batch_length=2048, batch_stride=16, **kwargs):

        if batch_length < 512:
            batch_length = 512

        encoded_input = tokenizer(input_text, padding="max_length", truncation=True, max_length=batch_length, stride=batch_stride, return_overflowing_tokens=True, add_special_tokens=False, return_tensors="pt")
        in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask
        gen_summaries = []
        for _id, _mask in zip(in_id_arr, att_arr):
            result, score = self.summarise_and_score(ids=_id, mask=_mask, model=model, tokenizer=tokenizer, **kwargs)
            score = round(float(score), 4)
            _sum = {"input_tokens": _id, "summary": result, "summary_score": score}
            gen_summaries.append(_sum)

        return gen_summaries

    def truncate_word_count(self, text, max_words=512):
        # split on whitespace with regex
        words = re.split(r"\s+", text)
        processed = {}
        if len(words) > max_words:
            processed["was_truncated"] = True
            processed["truncated_text"] = " ".join(words[:max_words])
        else:
            processed["was_truncated"] = False
            processed["truncated_text"] = text
        return processed
    
    def get_summary(self, input_text: str, model_size: str, num_beams, token_batch_length, length_penalty, repetition_penalty, no_repeat_ngram_size, max_input_length: int = 768):

        settings = {
            
        }
        st = time.perf_counter()
        history = {}
        clean_text = clean(input_text)#, lower=False)
        max_input_length = 1024 if model_size == "base" else max_input_length
        processed = self.truncate_word_count(clean_text, max_input_length)

        if processed["was_truncated"]:
            tr_in = processed["truncated_text"]
            msg = f"Input text was truncated to {max_input_length} words (based on whitespace)"
            history["WARNING"] = msg
        else:
            tr_in = input_text

        _summaries = self.summarise_via_tokenbatches(
            tr_in,
            self.summariser_model_sm if model_size == "base" else self.summariser_model,
            self.tokenizer_sm if model_size == "base" else self.summariser_tokenizer,
            batch_length=token_batch_length,
            length_penalty= float(length_penalty),
            repetition_penalty= float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            encoder_no_repeat_ngram_size=4,
            num_beams=int(num_beams),
            min_length=4,
            max_length=int(token_batch_length // 4),
            early_stopping=True,
            do_sample=False
        )
        
        sum_text = [s["summary"][0] for i, s in enumerate(_summaries)]
        sum_scores = [
            f"\n - Section {i}: {round(s['summary_score'],4)}"
            for i, s in enumerate(_summaries)
        ]

        # summary scores can be thought of as representing the quality of the summary.
        # less-negative numbers (closer to 0) are better.
        return sum_text, sum_scores

    def split_text(self, text: str) -> list:

        sentences = re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", text)
        sentence_batches = []
        temp_batch = []

        for sentence in sentences:
            temp_batch.append(sentence)
            if len(temp_batch) >= 2 and len(temp_batch) <= 3 or sentence == sentences[-1]:
                sentence_batches.append(temp_batch)
                temp_batch = []

        return sentence_batches
    
    def correct_text(self, text: str, checker, corrector, separator: str = " ") -> str:

        sentence_batches = self.split_text(text)
        corrected_text = []

        for batch in tqdm(
            sentence_batches, total=len(sentence_batches), desc="correcting text.."
        ):
            raw_text = " ".join(batch)
            results = checker(raw_text)

            if results[0]["label"] != "LABEL_1" or (results[0]["label"] == "LABEL_1" and results[0]["score"] < 0.9):
                corrected_batch = corrector(raw_text)
                corrected_text.append(corrected_batch[0]["generated_text"])
            else:
                corrected_text.append(raw_text)

        corrected_text = separator.join(corrected_text)
        return corrected_text

    def flan_grammar_correct(self, text: str):
        text = clean(text[:50000])
        return self.correct_text(text, self.grammar_detector, self.grammar_corrector)


    """
        internal mechanics and matrix[n] setup, ongoing
    """

    # data management and structure
    # [[S,K], {frame:imageprompt}]
    
    def update_dict(self, d: dict, new_keys: list[int], new_values: list[str]) -> dict:
        # update matrix[1] with new keys and values
        d.clear() # this procedure resets
        for i, key in enumerate(new_keys):
            d[key] = [str(new_values[i])+":5", str(self.tokens_to_exclude)+":-1"] if i < len(new_values) else '' # weights must never sum to 0
        return d


    def divide_long_prompts_by_x(self, _prompts, x):
        # split the strings in _prompts by dots and commas
        # divide them by len(x)
        delimiter = ",."
        split_y = []
        for s in _prompts:
            split_y.extend([x for x in re.split('([' + delimiter + '])', s) if x])
        total_length = len(split_y)
        sub_length = math.ceil(total_length / len(x))
        if total_length < len(x):
            x = x[:total_length]
        sub_lists = [split_y[i:i+sub_length] for i in range(0, len(split_y), sub_length)]
        final_lists = []
        for i, sub_list in enumerate(sub_lists):
            while len(sub_list) < sub_length and i < len(sub_lists) - 1:
                sub_list.extend(sub_lists[i+1][:sub_length-len(sub_list)])
                sub_lists[i+1] = sub_lists[i+1][sub_length-len(sub_list):]
            final_lists.append(sub_list)
        final_y = []
        for sub_list in final_lists:
            final_y.append(''.join(sub_list))
        result = []
        for i in range(len(x) - 1):
            current_result = final_y[i]
            result.append(current_result)
        return result
    
    def add_to_dict_from_array(self, values, keys):
        # populate m[2,2]
        for i in range(len(values)):
            self.data[1][keys[i]] = values[i]
    
    def add_to_dict_from_summaries(self, values, keys, start_index, increment):
        new_dict = {key: " " for key in keys}
        value_index = 0
        for i in range(start_index, len(keys), increment):
            if value_index >= len(values):
                break
            new_dict[keys[i]] = values[value_index]
            value_index += 1
        self.data[1] = new_dict
    
    def sequence_arrays(self, x, y):
        for i in x:
            last_y = y[-1] + 25 if len(y) > 0 else 0
            self.data[1]
            y.extend([last_y + i, last_y + i + 25])
        return y

