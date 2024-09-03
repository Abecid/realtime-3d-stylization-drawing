# Realtime 3D Stylization

### Setup
---
* Setup a venv if you'd like to isolate your project environment: ```python -m venv env```
  * activate on MacOS: ```source ./env/bin/activate```
  * activate on Windows: ```env\Scripts\activate```
* Nvidia users should install PyTorch using this command: ```pip install torch --extra-index-url https://download.pytorch.org/whl/cu121```
* Install the requirements: ```pip install -r requirements.txt```
* Run ui.py: ```python ui.py```

After you run ui.py, models should be downloaded automatically to the models directory. It might take a few minutes depending on your network.
Once the models are downloaded, gradio will print to the console the url where you can access the ui.

