

# luis arandas 17-08-2023
# simple timeline object, only standard libs

import os
import datetime
import subprocess
import array

# separate audio file processor

class AudioProcessing:
    audio_extensions = ['.wav']
    loaded_audio_files = []

    @staticmethod
    def get_info_ffmpeg(file_path: str):
        cmd = ["ffmpeg", "-i", file_path]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        lines = result.stderr.split('\n')
        sample_rate = None
        num_samples = None

        for line in lines:
            if 'Audio:' in line:
                parts = line.split(',')
                for part in parts:
                    if 'Hz' in part:
                        sample_rate = int(part.strip().split(' ')[0])

            if 'Duration:' in line:
                duration = line.split(',')[0].split(':')[1].strip()
                parts = duration.split(':')
                if len(parts) == 3:
                    h, m, s = parts
                    total_seconds = int(h)*3600 + int(m)*60 + float(s)
                    if sample_rate:
                        num_samples = int(total_seconds * sample_rate)
                else:
                    print(f"Unexpected duration format for file: {file_path}")

        return sample_rate, num_samples

    @staticmethod
    def get_info_wav(file_path: str):
        with open(file_path, 'rb') as f:
            if f.read(4) != b'RIFF':
                raise ValueError('Not a valid WAV file.')
            
            f.read(4) # File size
            
            if f.read(4) != b'WAVE':
                raise ValueError('Not a valid WAV file.')
            
            while True:
                chunk_id = f.read(4)
                chunk_size = int.from_bytes(f.read(4), byteorder='little')
                
                if chunk_id == b'fmt ':
                    audio_format = int.from_bytes(f.read(2), byteorder='little')
                    num_channels = int.from_bytes(f.read(2), byteorder='little')
                    sample_rate = int.from_bytes(f.read(4), byteorder='little')
                    byte_rate = int.from_bytes(f.read(4), byteorder='little')
                    block_align = int.from_bytes(f.read(2), byteorder='little')
                    bits_per_sample = int.from_bytes(f.read(2), byteorder='little')
                    f.read(chunk_size - 16)  # Skip any remaining bytes in this chunk
                    
                elif chunk_id == b'data':
                    num_frames = chunk_size // block_align
                    break
                
                else:
                    f.read(chunk_size)
        
        return sample_rate, num_frames


    @staticmethod
    def process_folder(audio_folder: str):
        for audio_file in os.listdir(audio_folder):
            if any(audio_file.endswith(ext) for ext in AudioProcessing.audio_extensions):
                full_path = os.path.join(audio_folder, audio_file)
                sample_rate, num_samples = AudioProcessing.get_info_wav(full_path)
                print(f"File: {audio_file}, Sample Rate: {sample_rate}, Number of Samples: {num_samples}")
                AudioProcessing.loaded_audio_files.append([audio_file, sample_rate, num_samples])
            else:
                print(f"Skipping file not supported extension: {audio_file}")
        return AudioProcessing.loaded_audio_files
    

    @staticmethod
    def trim_audio(input_file, output_file, desired_duration_seconds):
        with open(input_file, 'rb') as f:
            riff = f.read(4)
            if riff != b'RIFF':
                raise ValueError('Not a valid WAV file.')
            
            _ = f.read(4)
            wave_id = f.read(4)

            fmt_id = f.read(4)
            fmt_size = int.from_bytes(f.read(4), byteorder='little')
            fmt_content = f.read(fmt_size)

            num_channels = int.from_bytes(fmt_content[2:4], byteorder='little')
            sample_rate = int.from_bytes(fmt_content[4:8], byteorder='little')
            bits_per_sample = int.from_bytes(fmt_content[14:16], byteorder='little')
            dtype = 'h' if bits_per_sample == 16 else 'i'  # 'h' is for int16 and 'i' is for int32

            while True:
                chunk_id = f.read(4)
                chunk_size = int.from_bytes(f.read(4), byteorder='little')
                if chunk_id == b'data':
                    break
                f.read(chunk_size)

            data = array.array(dtype)
            data.fromfile(f, chunk_size // data.itemsize)

            desired_samples = int(desired_duration_seconds * sample_rate) * num_channels
            trimmed_data = data[:desired_samples]

        with open(output_file, 'wb') as out:
            out.write(riff)
            out.write((len(trimmed_data) * (bits_per_sample // 8) + 36).to_bytes(4, byteorder='little'))
            out.write(wave_id)
            out.write(fmt_id)
            out.write(fmt_size.to_bytes(4, byteorder='little'))
            out.write(fmt_content)
            out.write(b'data')
            out.write((len(trimmed_data) * (bits_per_sample // 8)).to_bytes(4, byteorder='little'))
            out.write(trimmed_data.tobytes())


    @staticmethod
    def convert_tts_to_16bit(directory):
        for audio_file in os.listdir(directory):
            if any(audio_file.endswith(ext) for ext in AudioProcessing.audio_extensions):
                if "_16bit" not in audio_file:
                    output_file_name = os.path.splitext(audio_file)[0] + "_16bit.wav"
                    output_path = os.path.join(directory, output_file_name)
                    if not os.path.exists(output_path):
                        input_path = os.path.join(directory, audio_file)
                        command = [
                            'ffmpeg',
                            '-i', input_path,
                            '-sample_fmt', 's16',
                            output_path
                        ]
                        subprocess.run(command)

    

    @staticmethod
    # roughly glues audio together, background with tts keyframe
    # implemented to 22k, 16-bit as illustration

    def mix_audios_with_background(background_file, inserts_dict, output_file, fps=25):
        dtype = 'h'  # Assume 16-bit PCM wav samples
        with open(background_file, 'rb') as bg:
            riff = bg.read(4)
            _ = bg.read(4)  # Skip file size
            wave_id = bg.read(4)

            fmt_id = bg.read(4)
            fmt_size = int.from_bytes(bg.read(4), byteorder='little')
            fmt_content = bg.read(fmt_size)

            while True:
                chunk_id = bg.read(4)
                chunk_size = int.from_bytes(bg.read(4), byteorder='little')
                if chunk_id == b'data':
                    break
                bg.read(chunk_size)

            bg_data = array.array(dtype)
            bg_data.fromfile(bg, chunk_size // bg_data.itemsize)

        sorted_inserts = sorted(inserts_dict.items(), key=lambda x: x[0])

        for frame, insert_file in sorted_inserts:
            insert_position_seconds = frame / fps

            with open(insert_file, 'rb') as ins:
                _ = ins.read(12)  # Skip RIFF header and WAVE label
                while True:
                    chunk_id = ins.read(4)
                    chunk_size_ins = int.from_bytes(ins.read(4), byteorder='little')
                    if chunk_id == b'data':
                        break
                    ins.read(chunk_size_ins)

                ins_data = array.array(dtype)
                ins_data.fromfile(ins, chunk_size_ins // ins_data.itemsize)

            insert_position_samples = int(insert_position_seconds * 22050)
            bg_data[insert_position_samples:insert_position_samples] = ins_data

        with open(output_file, 'wb') as out:
            out.write(riff)
            out.write((len(bg_data) * 2 + 36).to_bytes(4, byteorder='little'))  # 2 bytes/sample for 16-bit audio
            out.write(wave_id)
            out.write(fmt_id)
            out.write(fmt_size.to_bytes(4, byteorder='little'))
            out.write(fmt_content)
            out.write(b'data')
            out.write((len(bg_data) * 2).to_bytes(4, byteorder='little'))  # 2 bytes/sample for 16-bit audio
            bg_data.tofile(out)
    


class VirtualTimeline:
    def __init__(self, fps=25):
        self._timeline = {}
        self._current_frame = 0
        self._batch_name = ""
        self._max_frame = 50000 # 33.333 minutes
        self._audio_lengths = []
        self._total_audio_duration = 0
        self._fps = fps
        self._loaded_prompts = []
        self._prompts_to_exclude = []


    def get_timeline(self) -> dict[int, list[str]]:
        return self._timeline
    
    def get_fps(self) -> int:
        return self._fps
    

    def add_event(self, time_point: int, event: str):
        if time_point in self._timeline:
            self._timeline[time_point].append(event)
        else:
            self._timeline[time_point] = [event]

    
    
    def update_dict(self, d: dict, new_keys: list[int], new_values: list[str]) -> dict:
        d.clear() # this resets
        for i, key in enumerate(new_keys):
            exclude_str = f", {self._prompts_to_exclude}:-1" if self._prompts_to_exclude else ""
            d[key] = f"{new_values[i]}:5{exclude_str}" if i < len(new_values) else ''  # weights must never sum to 0
        return d
    

    def compute_total_frame_count(self) -> int:
        total_frames = sum([int(length * self._fps) for length in self._audio_lengths])
        return total_frames
    
    
    def get_files_from_directory(self, directory_path):
        return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    
    def timecode_to_seconds_frames(self, timecode: str) -> tuple:
        components = timecode.split(':')
        if len(components) == 4:
            h, m, s, _ = map(int, components)
        elif len(components) == 3:
            h, m, s = map(int, components)
        else:
            raise ValueError("Invalid timecode format")
        
        total_seconds = h * 3600 + m * 60 + s
        total_frames = total_seconds * 25
        return total_seconds, total_frames
    
    

    def create_frame_prompt_dictionary(self, frame_array: list, prompts: list) -> dict:
        frame_prompt_dict = {}
        for i, frame in enumerate(frame_array):
            if i < len(prompts):
                prompt = prompts[i]
            else:
                prompt = ""
            frame_prompt_dict[frame] = str(prompt)
        return frame_prompt_dict
    
    

    def compute_transition_frames_raw(self) -> list:
        frame_counter = 0
        transition_frames = []

        for length in self._audio_lengths:
            transition_frames.append(frame_counter)
            frame_counter += int(length * self._fps)
        return transition_frames



    def compute_transition_frames_1sec(self) -> list:
        frame_counter = self._fps  # Start with 1-second buffer
        transition_frames = [frame_counter]

        for length in self._audio_lengths[:-1]:
            frame_counter += int(length * self._fps)
            transition_frames.append(frame_counter)

        frame_counter += int(self._audio_lengths[-1] * self._fps)
        transition_frames.append(frame_counter + self._fps)
        transition_frames.insert(0, 0)
        transition_frames[-1] = transition_frames[-2] + 25
        return transition_frames
    



    def compute_transition_frames_5sec(self) -> list:
        frame_counter = self._fps  # Start with 1-second buffer
        transition_frames = [frame_counter]  

        for length in self._audio_lengths[:-1]: 
            frame_counter += int(length * self._fps)
            transition_frames.append(frame_counter)

        frame_counter += int(self._audio_lengths[-1] * self._fps)
        transition_frames.append(frame_counter + self._fps)
        transition_frames[1:] = [frame + 5 * self._fps for frame in transition_frames[1:]]
        transition_frames.insert(0, 0)
        transition_frames[-1] = transition_frames[-2] + 25

        return transition_frames



    def compute_input_equal_distance(self, timecode: str, prompts: list) -> list:
        _, variable_size = self.timecode_to_seconds_frames(timecode)  # Use self to reference
        prompt_interval = variable_size / (len(prompts) - 1) if len(prompts) > 1 else variable_size
        frame_array = []
        prompt_frame = 0
        for _ in prompts:
            frame_array.append(int(prompt_frame))
            prompt_frame += prompt_interval
        
        frame_array[-1] = int(variable_size)  # Update the last frame prompt element
        return frame_array
    
    
    def compute_total_duration(self, audio_files_info: list) -> int:
        total_duration_seconds = 0
        for audio_info in audio_files_info:
            _, sample_rate, num_samples = audio_info
            duration_seconds = num_samples / sample_rate
            total_duration_seconds += int(duration_seconds)
            self._audio_lengths.append(int(duration_seconds))
        self._total_audio_duration = total_duration_seconds
    

    # example sequencers

    def compute_sequencer_1(self, time_input: str, prompts: list):
        key_arr = self.compute_input_equal_distance(time_input, prompts)
        frame_prompt_dict = self.create_frame_prompt_dictionary(key_arr, prompts)
        self._loaded_prompts = prompts
        self._timeline = frame_prompt_dict
        if self._batch_name == "":
            self._batch_name = "None"
        self.export_timeline_to_txt(self._batch_name, time_input)



    def compute_sequencer_2(self, prompts: list):
        x = self.compute_transition_frames_1sec()
        prompts.insert(0, "")
        frame_prompt_dict = self.create_frame_prompt_dictionary(x, prompts)
        print("frame prompt dict ", frame_prompt_dict)
        self._loaded_prompts = prompts
        self._timeline = frame_prompt_dict
        if self._batch_name == "":
            self._batch_name = "None"
        self.export_timeline_to_txt(self._batch_name, "Derived")



    def compute_sequencer_3(self, prompts: list):
        x = self.compute_transition_frames_5sec()
        prompts.insert(0, "")
        frame_prompt_dict = self.create_frame_prompt_dictionary(x, prompts)
        print("frame prompt dict ", frame_prompt_dict)
        self._loaded_prompts = prompts
        self._timeline = frame_prompt_dict
        if self._batch_name == "":
            self._batch_name = "None"
        self.export_timeline_to_txt(self._batch_name, "Derived")


    def export_timeline_to_txt(self, batch_name: str, timecode: str, filename="timeline.txt"):
        current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_text = '\n'.join([f"{key}: {''.join(value) if value else 'Black.'}" for key, value in self._timeline.items()])
        tts_path = os.path.join(os.getcwd(), "tts_output/")
        audios = self.get_files_from_directory(tts_path)

        structured_content = (
            f"time: {current_timestamp}\n"
            f"length timecode: {timecode}\n"
            f"batch name: {batch_name}\n"
            f"max frame: {self._max_frame}\n"
            f"audio files: {audios}\n"
            "_____\n\n"
            f"{file_text}"
        )

        export_dir = os.path.join(os.getcwd(), "text_output")
        existing_files = [f for f in os.listdir(export_dir) if f.startswith("timeline_") and f.endswith(".txt")]
        
        if existing_files:
            max_number = max([int(f.split('_')[1].split('.')[0]) for f in existing_files])
            next_number = max_number + 1
        else:
            next_number = 1

        filename = f"timeline_{next_number:05}.txt"

        full_path = os.path.join(export_dir, filename)
        with open(full_path, 'w') as file:
            file.write(structured_content)

    
    def utils_extract_sentences_from_path(self, path):
        loaded_summaries = []

        def extract_sentences_from_file(filepath):
            with open(filepath, 'r') as file:
                content = file.read()
            sentences = content.split("_____\n")[1].split('.')
            sentences = [sentence.strip() for sentence in sentences if sentence.strip() and len(sentence.strip()) > 1]
            return sentences

        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(path, filename)
                    loaded_summaries.extend(extract_sentences_from_file(file_path))
        elif os.path.isfile(path):
            loaded_summaries.extend(extract_sentences_from_file(path))
        else:
            print(f"invalid path: {path}")

        return loaded_summaries
    
    def utils_load_timeline_from_file(filepath):
        with open(filepath, 'r') as file:
            content = file.read()

        timeline_content = content.split("_____\n\n")[1]
        timeline_dict = {}
        for line in timeline_content.strip().split('\n'):
            key, value = line.split(": ", 1)
            timeline_dict[int(key)] = value

        return timeline_dict



