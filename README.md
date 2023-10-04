### virtual-timeline clipguided

Small implementation of a language-based virtual timeline object, made to interface with CLIP-guided DDPM. Establishes a template for summarisation and organisation of events following multitrack methods; intends to coordinate diffusion renders with TTS. Environment developed with Ubuntu 22.04, nvidia 470.161 and CUDA 11.4.

```
(download libraries)
conda create --name vtc --file bio_env.txt && conda activate vtc
(compute summaries from input file)
! python3 bio_1.py --text_dataset /path/to/dataset.txt
(generates a file tree for the amount of summaries, pattern:)
_______________________________________________________________
> time: 2023-09-27 15:30:06
> id: 0
> dataset path: ./path/to/dataset.txt
> model used: model_name
> _____
> output_x
_______________________________________________________________
(compute new text from input summaries)
! python3 bio_2.py --summaries_dataset /path/to/dataset.txt
(compute TTS on newly-generated text)
! python3 bio_3.py --speech_from_text /path/to/dataset.txt
(compute timelines and export pattern to be read by audio encoder)
! python3 bio_4.py --defined_length --audio_folder path/to/tts_files --length 00:02:00:00
! python3 bio_4.py --derived_length --audio_folder path/to/tts_files --split_pad 1
! python3 bio_4.py --derived_length --audio_folder path/to/tts_files --split_pad 5
_______________________________________________________________
> time: 2023-09-27 15:30:06
> length timecode: 00:00:00:00
> batch name: None
> max frame: 50000
> audio files tts: []
> _____
> key: val
_______________________________________________________________
<compute DDPM from timeline params>
(encoding illustration TTS with background at prompt keyframe)
! python3 bio_5.py --timeline_file path/to/timeline.txt
(example usage)
! # timeline = VirtualTimeline()
! # timeline.add_event(100, 'a man is dancing in the snow')
! # timeline.compute_sequencer_1(time_input, prompts)
! # timeline.compute_sequencer_2(prompts)
! # timeline.compute_sequencer_3(prompts)
! # timeline.export_timeline_to_txt(batch_name, type)
```

```
@article{insamla23,
  title={Computing short films using language-guided diffusion and vocoding through virtual timelines of summaries},
  author={Arandas, Luís and Grierson, Mick and Carvalhais, Miguel},
  journal={INSAM, Journal of Contemporary Music, Art and Technology},
  year={2023}
}
```