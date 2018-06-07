# MedNLP
This repository contains code and analysi for ''Disease Identification and NLP answering chatbot for Discharge Summaries'' for MIMIC-III dataset.

The project is aimed at building a system that leverages huge healthcare information available as electronic data and learn a disease identification model. The project also focusses on creating a Human Computer Interaction component in the form of a NLP chatbot.

### Files
* MED277_bot.ipynb - A jupyter notebook containing code for NLP chatbot
* MED277_bot.py - A pyton script for initializing and running the chatbot. Contains same code as MED277_bot.ipynb

#### Running the chatbot [Complete Data]
The chatbot uses the data from MIMIC-III discharge summaries to train and answer questions. The proper data path of MIMIC-III dataset **NOTEEVENTS.csv.gz** should be set in *base_path*. Then navigate to the path of **MED277_bot.ipynb** run the jupyter notebook using the following command:
```python
jupyter notebook
```
This should start jupyter and then you can run the notebook **MED277_bot.ipynb**

To run the python file **MED277_bot.py**, the code for reading from the file should be uncommented, and the code to read from saved data file  **data10.pkl** should be commented in **load_data()** function. By default the code for reading data is commented out in the MED277_bot.py file. Then the file can be executed using the following command:
```python
python MED277_bot.py
```

#### Running the chatbot [Small Subset Data]
We have saved small subset of data to a pickle file **data10.pkl**. 
Just edit the *base_path* inside ***MED277_bot.py*** file **load_data()** function, and keep this file at the location of *base_path*.  The file ***MED277_bot.py*** should then run using the following command:
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

### Required Dependencies & Libraries
- anaconda 5.2
- Python 3.x
- pandas
- sklearn
- nltk
- numpy