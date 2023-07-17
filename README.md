

# virtual-timeline-clipguided

simple python implementation of a language-based virtual timeline  
establishes a simple template to compute video sequences and TTS through summarisation and further CLIP embeddings
environment developed with ubuntu 22.04, nvidia 470.161 and CUDA 11.4. runs on 24Gb at [720,405]; e.g. PNY XLR8 3090

camera trucks are automated and hardcoded with disco-diffusion implementation, GUI will come on a separate repository with movement templates. this is research code, will optimise for reproducibility next couple of months. templates don't necessarily mean determinism on the GPU with CLIP-guidance


### installation and sequence render

```
$ sudo chmod +x ./install.sh && source ./install.sh
(get environment from install)
$ python3 virtual_timeline_clipguided.py --render (batch_steps : int) (batch_cgs : int) (batch_skip_steps : int) (batch_fss : int) (cam_t_x : float) (cam_t_y : float) (cam_t_z : float) (cam_r_x : float) (cam_r_y : float) (cam_r_z : float)
(example, run sequence)
$ python3 virtual_timeline_clipguided.py --render 150 7500 50 50 0.0 0.0 0.3 0.0 0.0 0.0 
$ python3 virtual_timeline_clipguided.py --render 100 7500 50 64 0.0 0.0 0.0 0.2 0.0 0.0 
(soon -> namespace templates in text files)
```

### useful variables

```
timeline.fps = 25
timeline.film_length_secs = 20

# generate audio files
timeline.tts_prompts_by_sentence()

# generate text prompts to further CLIP embeddings
timeline.sequencer_1()
# timeline.sequencer_2()
# timeline.sequencer_3()
# timeline.sequencer_random()

# (useful commands) -> rescale the output as 16:9 1080p
$ mkdir output_folder && for file in /home/user/Desktop/folder/*.mp4; do ffmpeg -i "$file" -vf "scale=1920:1080" -c:v libx264 -crf 23 -c:a copy "output_folder/output_$(printf '%05d' $((i++))).mp4"; done
```

### citation

```
@article{insamla23,
  title={Computing short films using language-guided diffusion and vocoding through virtual timelines of summaries},
  author={Arandas, Luís and Grierson, Mick and Carvalhais, Miguel},
  journal={INSAM, Journal of Contemporary Music, Art and Technology},
  year={2023}
}
```

