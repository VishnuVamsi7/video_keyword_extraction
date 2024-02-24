from moviepy.editor import VideoFileClip

# Load the video file
video_clip = VideoFileClip('/content/drive/MyDrive/Colab Notebooks/SP_Project/Videos/Untitled video - Made with Clipchamp.mp4')

# Extract the audio
audio_clip = video_clip.audio

# Save the audio to a file
audio_clip.write_audiofile('output_audio.wav')

# Close the video and audio clips
video_clip.close()
audio_clip.close()
