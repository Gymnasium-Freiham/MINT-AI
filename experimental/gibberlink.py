# play_sound.py
import simpleaudio as sa
import sys

filename = sys.argv[1]
wave_obj = sa.WaveObject.from_wave_file(f"./{filename}")
play_obj = wave_obj.play()
play_obj.wait_done()
