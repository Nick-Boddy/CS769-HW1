Nick Boddy | 2024-09-19

HW1 Submission for CS 769


## REQUIRED FILES

My model parameters, as defined in run_exp.sh, require the GloVe embeddings.

It expects ./glove.6B/glove.6B.300d.txt to be available.

run_exp.sh will run my setup.py file which downloads and extracts the needed file to the needed location.


## IMPLEMENTATION DETAILS

NBoW with word dropout and embedding averaging, ff dropout, pre-trained embeddings incorporation, and weight initialization of xavier or kaiming, depending on activations.
The inspiration of the model is DANModel.


## PERFORMANCE

sst-test: 0.4421

sst-dev: 0.4460

cfimdb-test: 0.5000

cfimdb-dev: 0.9306