# MedNLP
This repository contains code and analysis for ''NLP chatbot for Discharge Summaries'' using MIMIC-III dataset.

The project is aimed at building a system that leverages huge healthcare information available as electronic data and focusses on creating a Human Computer Interaction component in the form of a NLP chatbot. This system answers questions about a patient's discharge summaries primarily based on topic modeling and matching techniques.

### Files
* MED277_bot.ipynb - A jupyter notebook containing code for NLP chatbot.
* MED277_bot.py - A pyton script for initializing and running the chatbot. Contains same code as MED277_bot.ipynb.
* data10.pkl - A subset of 50 discharge summaries extracted from MIMIC-II dataset and saved as a python pickle file. The file can be found at [location](https://drive.google.com/open?id=19-Wh4x-roinUDStfiZ_C2BmmcCr4UVNY)
* MED277_report.pdf - A pdf report for the project.
* MED277_presentation.pptx - A power presentation for the project.

#### Running the chatbot [Complete Data]
The chatbot uses the data from MIMIC-III discharge summaries to train and answer questions. The proper data path of MIMIC-III dataset *NOTEEVENTS.csv.gz* should be set in *base_path*. Then navigate to the path of **MED277_bot.ipynb** run the jupyter notebook using the following command:
```python
jupyter notebook
```
This should start jupyter and then you can run the notebook **MED277_bot.ipynb**

To run the python file **MED277_bot.py**, the code for reading from the file should be uncommented and proper file *NOTEEVENTS.csv.gz* downloaded from MIMIC-III dataset should be kept at the *base_path* location, and the code to read from saved data file  **data10.pkl** should be commented in **load_data()** function. By default the code for reading data is commented out in the MED277_bot.py file. Then the file can be executed using the following command:
```python
python MED277_bot.py
```

#### Running the chatbot [Small Subset Data]
We have saved small subset of data to a pickle file **data10.pkl**. 
Just edit the *base_path* inside ***MED277_bot.py*** file **load_data()** function, and keep this file at the location of *base_path*.  The file ***MED277_bot.py*** should then run using the following command:
```python
python MED277_bot.py
```
Follow the on screen instructions for interacting further with chatbot.

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