import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")
volume = engine.getProperty("volume")
engine.setProperty("volume",0.1)
engine.setProperty("voice",voices[1].id)
engine.say("sanskar is a noob")
engine.runAndWait()