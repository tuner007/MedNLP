# MedNLP
This repository contains code and analysi for ''Disease Identification and NLP answering chatbot for Discharge Summaries'' for MIMIC-III dataset.

The project is aimed at building a system that leverages huge healthcare information available as electronic data and learn a disease identification model. The project also focusses on creating a Human Computer Interaction component in the form of a NLP chatbot.

### Files
* MED277_bot.ipynb - A jupyter notebook containing code for NLP chatbot
* MED277_bot.py - A pyton script for initializing and running the chatbot. Contains same code as MED277_bot.ipynb

### Running the chatbot
The chatbot uses the data from MIMIC-III discharge summaries to train and answer questions. The proper data path should be set in *base_path* and the code for reading from the file should be uncommented in **load_data()** function. By default the code for reading data is commented out in the MED277_bot.py file.
```python
python MED277_bot.py
```