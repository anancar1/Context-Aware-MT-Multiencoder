# Context-Aware-MT-Multiencoder
**1- Create a new virtual environment in python and activate it**

python3.7 -m venv venvContext

source venvContext/bin/activate

**2- Install the required libraries for executing scripts correctly**

pip install --upgrade setuptools

pip install --upgrade pip

pip install -r requeriments.txt


**3- Create Subword text encoder for source and target languages (this process usually takes 10 min):**

python extract_vocab.py -p ./data/ -s es -t en

**4- Train the sentece-level Transformer (30 epochs)**

python train.py -p ./data/ -s es -t en -checkpoint ./chekcpoints/ 

**5- Train the multiencoder context aware Trasnformer(5 epochs)**

python train.py -p ./data/ -s es -t en -checkpoint ./checkpoints/ --context

**6- Evaluate sentence-level Transformer**

python evaluate.py -p ./data/ -s es -t en -checkpoints ./checkpoints/

**7- Evaluate context-aware Transformer**

python evaluate.py -p ./data/ -s es -t en -checkpoints ./checkpoints/ --context
