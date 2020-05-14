# Sequence Labeling using CRF
## Description
Sequence Labeling is very useful in many NLP applications, including conversational assistants like [Alexa](https://s3.us-east-2.amazonaws.com/alexapapers/NER-adaptation-AAAI2019-Moschitti.pdf).
I built two sequence label taggers using [Conditional Random Fields](https://medium.com/@phylypo/nlp-text-segmentation-using-conditional-random-fields-e8ff1d2b6060) and trained them on the
[Switchboard DMSL](https://web.stanford.edu/~jurafsky/ws97/manual.august1.html) data. The difference between the two taggers being improved feature set.
I saw a jump in performance from 71% to 78%, using [bi-grams](https://en.wikipedia.org/wiki/Bigram) of text.

### Built With
* [Python 3](https://www.python.org/downloads/release/python-382/)
* [Python CRFSuite](https://python-crfsuite.readthedocs.io/en/latest/)
* [Some knowledge on Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field)
* [Some research on Feature Extraction](https://www.researchgate.net/profile/Shamila_Nasreen/publication/265727419_A_Survey_Of_Feature_Selection_And_Feature_Extraction_Techniques_In_Machine_LearningSAI2014/links/55b12f2c08aec0e5f4310e76.pdf)

## Getting Started
To get a local copy up and running follow these simple example steps.
### Pre-requisites
* Install Python. Detailed instructions for installation can be found [here](https://realpython.com/installing-python/).
* Mac users, with homebrew can run the following command in their terminal.
```sh
brew install python3
```
* While I can give the command for conda users to install python, if you're running conda, you'd probably have it. You can check the version to make sure it is Python 3.
```sh
python --version
```
* Install pycrfsuite from [here](https://python-crfsuite.readthedocs.io/en/latest/)

### Installation
* Clone this project using the following command.
```
git clone https://github.com/Narasimhag/SequenceLabelingWithCRF
```

### Usage
1. Download the data from the DAMSL link above and extract it to the project directory, created after running the clone command, typically named 'SequenceLabelingWithCRF'.
2. Divide the data into training and testing sets.
3. Run the baseline_tagger.py as follows.
```sh
python3 path/to/baseline_tagger.py /path/to/training/data /path/to/output/data /path/to/outputfile
```
4. Run the advanced_tagger.py as follows.
```sh
python3 path/to/advanced_tagger.py /path/to/training/data /path/to/output/data /path/to/outputfile
```
5. A small accuracy check code prints the accuracies to the terminal. You can observe advanced_tagger outperforms the baseline_tagger.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Contact

If you have some criticism or want to say some nice things about the project, please feel free to tweet me. @raogundavarapu or email me at raonarasimha050@gmail.com
