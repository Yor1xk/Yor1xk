# sign-language-detector-python

The project was created by [Computer Vision Engineer](https://github.com/computervisioneng).

[![Author's Youtube Channel and video of the project](https://img.youtube.com/vi/MJCSjXepaAM/0.jpg)](https://www.youtube.com/watch?v=MJCSjXepaAM)


I extended the project's base functionality to be able to detect multiple hands and correctly classify them without crashing, since, after some testing, I've noticed that when the model tried to classify multiple hands it would crash.

The project uses Mediapipe's Hands solution, Scikit-learn and OpenCV; these three libraries are the libraries that need to be installed in order for the project to work.

[!TIP]
The libraries and their correspective versions are contained in the `requirements.txt`

To install the required libraries you need to run `pip install -r requirements.txt` in the terminal/console while you are in the folder where the project has been installed.
If a problem occurs during installation process, try changing Python's interpreter version or changing the version of the libraries in the `requirements.txt`.

To run the project, you need to connect a webcam to your device. You might need to change the index of the webcam(default is 0)

In order to make the project work, you need to run following scripts in this exact order:
1. `collect_imgs.py`
2. `create_dataset.py`
3. `train_classifier.py`
4. `inference_classifier.py`

The project supports recognition of 3 characters. To increase the amount of recognized letters, you need to add them manually by modifying code. The variables that need to be modified are the labels_dict in `inference_classifier.py` and number of classes in the `collect_imgs.py`.








