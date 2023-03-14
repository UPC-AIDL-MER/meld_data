import os
import pickle
import torch
import torchaudio
import torchvision.io as tio
import torchvision.utils as tu

class MeldAudioPreprocess():
    def __init__(self, meld_raw_path, preprocess_output_path):
        self.split_folders = { 
            "train": "train_splits", 
            "test": "output_repeated_splits_test", 
            "dev": "dev_splits_complete"
        }

        self.meld_raw_path = meld_raw_path
        self.preprocess_output_path = preprocess_output_path

        self.silence_constant = torch.tensor(-100.0)
        self.min_inf_constant = torch.tensor(float("-inf"))

        for key, value in self.split_folders.items():
            split_path = os.path.join(self.meld_raw_path, value)
            if not os.path.exists(split_path):
                print(f"ERROR: split folder {split_path} doesn't exist")

    def __extractFromVideo(self, save_wav=False, save_img=False):
        for key, value in self.split_folders.items():
            split_path = os.path.join(self.meld_raw_path, value)
            preprocessed_path = os.path.join(self.preprocess_output_path, value)
            data_dictionary = {}

            if not os.path.exists(preprocessed_path):
                try:
                    os.makedirs(preprocessed_path)
                except:
                    print(f"ERROR: couldn't create directory {preprocessed_path}")
                    exit()

            for file in os.listdir(split_path):
                input_file = os.path.join(split_path, file)

                extracted_jpg_file = os.path.join(preprocessed_path, file + '.jpg')
                extracted_wav_file = os.path.join(preprocessed_path, file + '.wav')

                # Handle special case of test split naming conventions
                if "final_videos_test" in file:
                    file = file.replace("final_videos_test", "")

                video, audio, info = tio.read_video(input_file)
                wav_left = audio[0,:] # Use first channel only
                source_sample_rate = info['audio_fps']
                
                if save_wav:
                    torchaudio.save(extracted_wav_file, audio, source_sample_rate)
                
                target_sample_rate = 8000
                resampler = torchaudio.transforms.Resample(source_sample_rate, target_sample_rate, dtype=wav_left.dtype)
                resampled_wav_left = resampler(wav_left)
                win_length_samples = int(0.020 * target_sample_rate)
                n_fft = 1024
                n_mels = 40

                print(f'Saving spectrogram data for {file}')
                data_dictionary[file] = self.__saveSpectrogram(extracted_jpg_file, resampled_wav_left, target_sample_rate, n_fft, win_length_samples, n_mels, save_img)
        
            self.__savePreprocessedData(data_dictionary, key)

    def __saveSpectrogram(self, jpg_file, audio, sample_rate, n_fft, win_length, n_mels, save_img=False):
        try:
            transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, n_mels=n_mels, center=True)
            mel_spectrogram = transform(audio)
            mel_spectrogram_log = torch.log(mel_spectrogram)
            mel_spectrogram_log[mel_spectrogram_log == self.min_inf_constant] = self.silence_constant

            if save_img:
                tu.save_image(mel_spectrogram_log, jpg_file)

            return mel_spectrogram_log
        except: # If there is not enough data or if it's corrupted, return a tensor of one frame with silence values
            min_tensor = torch.zeros(n_mels, 1)
            min_tensor.fill_(self.silence_constant)
            return min_tensor

    def __savePreprocessedData(self, data, split_name):
        out_pickle_file = os.path.join(self.preprocess_output_path, f'meldaudiodata_{split_name}.pickle')

        with open(out_pickle_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved split data at {out_pickle_file}')

    def preprocess(self):
        self.__extractFromVideo()

if __name__ == "__main__":
    meld_preprocess = MeldAudioPreprocess("..\MELD.Raw.tar", "..\MELDAudioPreprocessed")
    meld_preprocess.preprocess()
