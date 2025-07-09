import whisper
import ssl
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import librosa
import numpy as np
from textblob import TextBlob
import subprocess
import nltk
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

class AudioAnalysis:
    def __init__(self):        
        self.filler_words = {
            "um", "uh", "like", "you know", "so", "actually", "basically",
            "seriously", "literally", "i mean", "you know", "kind of", "sort of",
            "well", "actually", "right", "actually", "basically", "i guess",
            "honestly", "to be honest", "basically", "you see", "in a way",
            "for real", "in fact", "just", "okay", "pretty much", "like i said",
            "at the end of the day", "as i was saying", "the thing is", "to tell the truth",
            "believe me", "for sure", "more or less", "in other words", "let me think",
            "give me a second", "hold on", "let's see", "i don't know", "let me see",
            "that's the thing", "i mean, i don't know"
        }

        self.speechModel = self.loadSpeechModel()
        self.pauseModel = self.loadPauseModel()
        self.sentimentModel = self.loadSentimentModel()
        nltk.download("punkt_tab")
        nltk.download('averaged_perceptron_tagger_eng')
        print("Loaded Speech Models")

    def extractAudio(self, output_audio_path: str, input_video_path: str):
        command = [
            'ffmpeg',  
            '-y',
            '-i', input_video_path, 
            '-vn', 
            '-acodec', 'libmp3lame', 
            '-q:a', '0',  
            output_audio_path 
        ]
        
        try:
            subprocess.run(command, check=True)
            print(f"Audio extracted successfully and saved to {output_audio_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")

        return output_audio_path


    def loadSpeechModel(self):
        try:
            model = whisper.load_model("base.en")
            return model
        except Exception as e:
            print(f"Error loading speech model: {e}")
            return None

    def loadPauseModel(self):
        try:
            model = load_silero_vad()
            return model
        except Exception as e:
            print(f"Error loading pause model: {e}")
            return None

    def loadSentimentModel(self):
        try:
            model = SentimentIntensityAnalyzer()
            return model
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            return None

    def loadAudioFile(self, audioFilePath):
        try:
            y, sr = librosa.load(audioFilePath, sr=None, duration=None)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

    def getDuration(self, y):
        try:
            return librosa.get_duration(y=y)
        except Exception as e:
            print(f"Error calculating audio duration: {e}")
            return 0

    def getTranscription(self, audioFilePath):
        try:
            result = self.speechModel.transcribe(audioFilePath)
            if result and result.get("text"):
                return result["text"]
        except Exception as e:
            print(f"Error transcribing audio: {e}")
        return None

    def extractPauses(self, audioFilePath, long_pause_threshold=1.8, short_pause_min=0.5, short_pause_max=1.8):
        try:
            wav = read_audio(audioFilePath)
            segments = get_speech_timestamps(
                wav,
                self.pauseModel,
                return_seconds=True,
            )

            long_pauses = []
            short_pauses = []

            for i in range(len(segments) - 1):
                end_time = segments[i]["end"]
                start_time_next = segments[i + 1]["start"]

                pause_duration = start_time_next - end_time

                if pause_duration > long_pause_threshold:
                    long_pauses.append((end_time, start_time_next, pause_duration))
                elif short_pause_min <= pause_duration <= short_pause_max:
                    short_pauses.append((end_time, start_time_next, pause_duration))

            return len(long_pauses), len(short_pauses)
        except Exception as e:
            print(f"Error extracting pauses: {e}")
            return 0, 0

    def getSentiment(self, sentence):
        try:
            sentiment_dict = self.sentimentModel.polarity_scores(sentence)
            return {
                "Negative Sentiment Score": sentiment_dict["neg"] * 100,
                "Neutral Sentiment Score": sentiment_dict["neu"] * 100,
                "Positive Sentiment Score": sentiment_dict["pos"] * 100,
                "Overall Sentiment Score": "Positive" if sentiment_dict["compound"] >= 0.05 else "Negative" if sentiment_dict["compound"] <= -0.05 else "Neutral"
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {}

    def extractPitchVariationRange(self, y, sr, hop_length=512, duration=None):
        try:
            pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, hop_length=hop_length)
            f0_indices = np.argmax(magnitudes, axis=0)
            f0 = pitches[f0_indices, np.arange(pitches.shape[1])]
            f0 = f0[f0 > 0]
            return f0, np.std(f0), np.ptp(f0)
        except Exception as e:
            print(f"Error extracting pitch variation range: {e}")
            return [], 0, 0

    def extractEnergy(self, y, hop_length=512, duration=None):
        try:
            rms = librosa.feature.rms(y=y, hop_length=hop_length)
            return rms[0]
        except Exception as e:
            print(f"Error extracting energy: {e}")
            return []

    def extractFormants(self, y, sr, hop_length=512, duration=None):
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
            f1 = mfcc[0]  
            f2 = mfcc[1] 
            return f1, f2
        except Exception as e:
            print(f"Error extracting formants: {e}")
            return [], []

    def extractZCR(self, y, hop_length=512):
        try:
            zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
            return zcr[0]
        except Exception as e:
            print(f"Error extracting ZCR: {e}")
            return []

    def extractHNR(self, y):
        try:
            hnr = librosa.effects.harmonic(y=y)
            return hnr
        except Exception as e:
            print(f"Error extracting HNR: {e}")
            return []

    def extractSpeechTempo(self, y, sr, audio_duration, hop_length=512, duration=None):
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
            syllables_count = len(onset_frames)
            tempo = syllables_count / audio_duration
            return tempo
        except Exception as e:
            print(f"Error extracting speech tempo: {e}")
            return 0

    def getWordMetrics(self, transcription, audio_duration):
        try:
            blob = TextBlob(transcription)
            words = blob.words
            word_count = len(words)

            total_minutes = audio_duration / 60
            avg_words_per_minute = word_count / total_minutes if total_minutes > 0 else 0

            unique_words = set(words)
            unique_words_per_minute = len(unique_words) / total_minutes if total_minutes > 0 else 0

            unique_word_count = len(unique_words)

            filler_words = [word for word in words if word.lower() in self.filler_words]
            filler_words_per_minute = len(filler_words) / total_minutes if total_minutes > 0 else 0

            nouns = adjectives = verbs = 0
            for sentence in blob.sentences:
                tags = sentence.tags
                nouns += sum(1 for word, pos in tags if pos in ["NN", "NNS", "NNP", "NNPS"])
                adjectives += sum(1 for word, pos in tags if pos in ["JJ", "JJR", "JJS"])
                verbs += sum(1 for word, pos in tags if pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])

            return {
                "Average Words Spoken Per Minute": avg_words_per_minute,
                "Average Unique Words Spoken Per Minute": unique_words_per_minute,
                "Unique Words Count": unique_word_count,
                "Filler Words Per Minute": filler_words_per_minute,
                "Total Nouns": nouns,
                "Total Adjectives": adjectives,
                "Total Verbs": verbs
            }
        except Exception as e:
            print(f"Error extracting word metrics: {e}")
            return {}
        
    def analyzeAudio(self, audioFilePath: str):
        try:
            y, sr = self.loadAudioFile(audioFilePath)
            audio_duration = self.getDuration(y=y)
            transcription = self.getTranscription(audioFilePath)
            f0, pitch_variation, pitch_range = self.extractPitchVariationRange(y, sr)
            energy = self.extractEnergy(y)
            f1, f2 = self.extractFormants(y, sr)
            zcr = self.extractZCR(y)
            hnr = self.extractHNR(y)
            speech_tempo = self.extractSpeechTempo(y, sr, audio_duration)
            pauses = self.extractPauses(audioFilePath)
            word_metrics = self.getWordMetrics(transcription, audio_duration)
            wpm = word_metrics["Average Words Spoken Per Minute"]
            sentiment = self.getSentiment(transcription)

            return {
                "f0": f0,
                "wpm": wpm,
                "energy": energy,
                "f1_f2": [f1, f2],
                "zcr": zcr,
                "hnr": hnr,
                "pitch_variation": pitch_variation,
                "speech_tempo": speech_tempo,
                "pauses": pauses,
                "pitch_range": pitch_range,
                "transcription": transcription,
                "word_metrics": word_metrics,
                "audio_duration": audio_duration,
                "sentiment": sentiment,
                "Average Unique Words Spoken Per Minute": round(word_metrics["Average Unique Words Spoken Per Minute"]),
                "Unique Words Count": round(word_metrics["Unique Words Count"]),
                "Filler Words Per Minute": round(word_metrics["Filler Words Per Minute"]),
                "Total Nouns": round(word_metrics["Total Nouns"]),
                "Total Adjectives": round(word_metrics["Total Adjectives"]),
                "Total Verbs": round(word_metrics["Total Verbs"])
            }

        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return {}        
    
    def compareAgainstBaseline(self, audio_metrics_comparison_plot, baseline_metrics_list, current_metrics):
        try:
            if not isinstance(baseline_metrics_list, list):
                baseline_metrics_list = [baseline_metrics_list]

            compare_keys = [
                "wpm", "speech_tempo", "pitch_variation", "pitch_range",
                "Average Unique Words Spoken Per Minute", "Unique Words Count",
                "Filler Words Per Minute", "Total Nouns", "Total Adjectives", "Total Verbs",
                "Long Pauses", "Short Pauses"
            ]

            def extract_pause_value(metrics, index):
                pauses = metrics.get("pauses", (0, 0))
                return pauses[index] if isinstance(pauses, tuple) else 0

            averaged_baseline = {}
            for key in compare_keys:
                if key == "Long Pauses":
                    values = [extract_pause_value(metrics, 0) for metrics in baseline_metrics_list]
                elif key == "Short Pauses":
                    values = [extract_pause_value(metrics, 1) for metrics in baseline_metrics_list]
                else:
                    values = [metrics.get(key, 0) for metrics in baseline_metrics_list]
                averaged_baseline[key] = mean(values) if values else 0

            comparison_results = {}
            inferences = []

            for key in compare_keys:
                baseline_val = averaged_baseline[key]
                if key == "Long Pauses":
                    current_val = extract_pause_value(current_metrics, 0)
                elif key == "Short Pauses":
                    current_val = extract_pause_value(current_metrics, 1)
                else:
                    current_val = current_metrics.get(key, 0)

                change = current_val - baseline_val
                direction = "up" if change > 0 else "down" if change < 0 else "no change"

                comparison_results[key] = {
                    "baseline": float(round(baseline_val, 2)),
                    "current": float(round(current_val, 2)),
                    "change": float(round(change, 2)),
                    "direction": direction
                }
            
                comparison_results["Overall Sentiment"] = current_metrics["sentiment"]["Overall Sentiment Score"]
                # comparison_results["Transcription"] = current_metrics["transcription"]

                # Inference rules
                if direction != "no change":
                    if key == "wpm":
                        inferences.append(f"Speaking rate went {direction}, which might indicate a shift in confidence or nervousness.")
                    elif key == "Filler Words Per Minute":
                        inferences.append(f"Filler words went {direction}, suggesting {'less clarity' if direction == 'up' else 'better fluency'}.")
                    elif key == "pitch_range":
                        inferences.append(f"Pitch range {direction}, possibly indicating {'higher expressiveness' if direction == 'up' else 'a flatter tone'}.")
                    elif key == "speech_tempo":
                        inferences.append(f"Speech tempo is {direction}, reflecting a potential change in energy or urgency.")
                    elif key == "Total Adjectives":
                        inferences.append(f"Adjective use is {direction}, which might reflect more descriptive or expressive language.")
                    elif key == "Average Unique Words Spoken Per Minute":
                        inferences.append(f"Lexical diversity per minute has gone {direction}, indicating vocabulary use is {'richer' if direction == 'up' else 'simpler'}.")
                    elif key == "Long Pauses":
                        inferences.append(f"Long pauses went {direction}, which could imply {'more hesitation or cognitive load' if direction == 'up' else 'smoother delivery'}.")
                    elif key == "Short Pauses":
                        inferences.append(f"Short pauses are {direction}, potentially indicating shifts in rhythm or phrasing.")

            # Plotting percent change for all metrics
            baseline_vals = [averaged_baseline[key] for key in compare_keys]
            current_vals = [
                extract_pause_value(current_metrics, 0) if key == "Long Pauses"
                else extract_pause_value(current_metrics, 1) if key == "Short Pauses"
                else current_metrics.get(key, 0)
                for key in compare_keys
            ]

            percent_changes = []
            for baseline, current in zip(baseline_vals, current_vals):
                if baseline == 0:
                    percent_changes.append(0)
                else:
                    percent_changes.append(((current - baseline) / baseline) * 100)

            pretty_labels = [
                key.replace("_", " ").title() if "_" in key else key.title()
                for key in compare_keys
                ]

            plt.figure(figsize=(12, 6))
            sns.set(style="whitegrid")
            colors = ["green" if change >= 0 else "red" for change in percent_changes]

            sns.barplot(x=pretty_labels, y=percent_changes, palette=colors)
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.ylabel("Percent Change from Baseline")
            plt.title("Relative Change in Audio Metrics (All)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(audio_metrics_comparison_plot)

            return {
                "comparison": comparison_results,
                "inferences": inferences
            }

        except Exception as e:
            print(f"Error comparing against baseline: {e}")
            return {}
