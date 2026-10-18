# Automatic Slide Generator - Dillon Babor

## The goal of this project is to take in organized human information and output a powerpoint presentation using gemini and sciduet.

## Need
* Python 3.10(+)
* Gemini API key
* Hugging Face access key (opt, makes it faster)

## SETUP

1) clone repository
2) install dependancies through pip
    pip install -r requirements.txt

3) set your keys in the terminal
    export GEMINI_API_KEY="insert-token-here"
    export HF_TOKEN="insert-token-here"

4) execute
    python pptxGen.py

## In your terminal you will see the rouge score and sentence transformation score. generated_pptx.pptx will contain the powerpoint generated from hugging faces sciduet dataset. summary_bullets.txt will be from the prehuman created bullet points.