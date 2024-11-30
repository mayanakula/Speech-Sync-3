# without background noise cancellation
#import speech_recognition as sr # type: ignore

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Function to capture live speech and convert it to text
# def recognize_speech_from_mic():
#     with sr.Microphone() as source:
#         print("Listening...")
#         # Adjust the recognizer sensitivity to ambient noise levels
#         recognizer.adjust_for_ambient_noise(source)
#         # Capture the audio
#         audio = recognizer.listen(source)
        
#         try:
#             # Recognize speech using Google Web Speech API
#             text = recognizer.recognize_google(audio)
#             print("You said: " + text)
#         except sr.RequestError:
#             print("API unavailable")
#         except sr.UnknownValueError:
#             print("Unable to recognize speech")

# if __name__ == "__main__":
#     recognize_speech_from_mic()
###with background noise cancellation
# import speech_recognition as sr
# import noisereduce as nr
# import numpy as np
# import io

# # Initialize the recognizer
# recognizer = sr.Recognizer()

# # Function to capture live speech, reduce noise, and convert it to text
# def recognize_speech_from_mic():
#     with sr.Microphone() as source:
#         print("Listening... Please speak into the microphone.")
#         # Adjust the recognizer sensitivity to ambient noise levels
#         recognizer.adjust_for_ambient_noise(source)
#         # Capture the audio
#         audio = recognizer.listen(source)

#         try:
#             # Convert the audio to a NumPy array
#             audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
#             # Perform noise reduction
#             reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)
#             # Convert the NumPy array back to audio data
#             reduced_noise_audio_io = io.BytesIO()
#             reduced_noise_audio_io.write(reduced_noise_audio.tobytes())
#             reduced_noise_audio = sr.AudioData(reduced_noise_audio_io.getvalue(), source.SAMPLE_RATE, 2)

#             # Recognize speech using Google Web Speech API
#             text = recognizer.recognize_google(reduced_noise_audio)
#             print("You said: " + text)
#         except sr.RequestError:
#             print("Could not request results; check your network connection.")
#         except sr.UnknownValueError:
#             print("Could not understand the audio.")

# if __name__ == "__main__":
#     recognize_speech_from_mic()
###With Google translation
# import speech_recognition as sr
# import noisereduce as nr
# import numpy as np
# import io
# from googletrans import Translator

# # Initialize the recognizer and translator
# recognizer = sr.Recognizer()
# translator = Translator()

# # Function to capture live speech, reduce noise, recognize speech, and translate text
# def recognize_and_translate_speech():
#     with sr.Microphone() as source:
#         print("Listening... Please speak into the microphone.")
#         # Adjust the recognizer sensitivity to ambient noise levels
#         recognizer.adjust_for_ambient_noise(source)
#         # Capture the audio
#         audio = recognizer.listen(source)
        
#         try:
#             # Convert the audio to a NumPy array
#             audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
#             # Perform noise reduction
#             reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)
#             # Convert the NumPy array back to audio data
#             reduced_noise_audio_io = io.BytesIO()
#             reduced_noise_audio_io.write(reduced_noise_audio.tobytes())
#             reduced_noise_audio = sr.AudioData(reduced_noise_audio_io.getvalue(), source.SAMPLE_RATE, 2)
            
#             # Recognize speech using Google Web Speech API
#             text = recognizer.recognize_google(reduced_noise_audio)
#             print("You said: " + text)
            
#             # Translate the recognized text
#             translated_text = translator.translate(text, dest='en')  # Translate to Spanish ('en'), can change to any language code
#             print("Translated text: " + translated_text.text)
#         except sr.RequestError:
#             print("Could not request results; check your network connection.")
#         except sr.UnknownValueError:
#             print("Could not understand the audio.")
#         except Exception as e:
#             print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     recognize_and_translate_speech()
import streamlit as st
import sounddevice as sd
import numpy as np
import io
from googletrans import Translator
from gtts import gTTS
import os
from textblob import TextBlob
import speech_recognition as sr
import noisereduce as nr

# Initialize translator
translator = Translator()

# Streamlit App Title
st.title("Speech Recognition & Translation Tool")

# Placeholder for microphone access status and outputs
status_placeholder = st.empty()
transcribed_text_placeholder = st.empty()
translated_text_placeholder = st.empty()

# Function to recognize and translate speech
def recognize_and_translate_speech():
    recognizer = sr.Recognizer()

    # Use sounddevice to record audio
    st.info("Listening... Please speak into the microphone.")
    try:
        # Set parameters for recording
        sample_rate = 16000  # Sample rate in Hz
        duration = 10  # Maximum duration in seconds

        # Record audio using sounddevice
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to finish

        # Reduce noise from audio
        reduced_noise_audio = nr.reduce_noise(y=audio_data.flatten(), sr=sample_rate)
        reduced_noise_audio_io = io.BytesIO()
        reduced_noise_audio_io.write(reduced_noise_audio.tobytes())
        reduced_noise_audio = sr.AudioData(reduced_noise_audio_io.getvalue(), sample_rate, 2)

        # Recognize speech
        text = recognizer.recognize_google(reduced_noise_audio)
        transcribed_text_placeholder.success(f"Transcribed Text: {text}")
        
        # Translate text to English
        translated_text = translator.translate(text, dest="en")
        translated_text_placeholder.success(f"Translated Text: {translated_text.text}")

        # Speak out the translated text using gTTS
        tts = gTTS(translated_text.text, lang='en')
        tts.save("output.mp3")
        st.audio("output.mp3", format='audio/mp3')

        # Analyze sentiment
        detect_and_express_emotion(translated_text.text)

    except sr.RequestError:
        status_placeholder.error("Error: Could not request results. Check your network connection.")
    except sr.UnknownValueError:
        status_placeholder.warning("Error: Could not understand the audio.")
    except Exception as e:
        status_placeholder.error(f"An error occurred: {e}")

# Function to detect and express emotion
def detect_and_express_emotion(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    st.info(f"Detected Sentiment: {sentiment}")
    if sentiment.polarity > 0:
        st.info("It sounds like you're feeling positive!")
    elif sentiment.polarity < 0:
        st.info("It sounds like you're feeling negative.")
    else:
        st.info("It sounds like you're feeling neutral.")

# Streamlit interface for capturing user actions
if st.button("Start Listening"):
    recognize_and_translate_speech()
