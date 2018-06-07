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

### Sample questions for chatbot
- What is my date of birth?
- What is my admission date?
- When was I discharged?
- What is my gender?
- What are the services I had?
- Do I have allergy?
- Who was my attending?
- Am I married?
- What is my social history?
- How can I make an appointment?
- Do I need to visit the clinic?
- How was my MRI?
- What are the medication I should take?
- How to take the steroid?
- What do I do if I have seizures?
- Is my vision blurry?
- Is something wrong with my brain?
- Do I have a cold?
- Do I have dysphagia?