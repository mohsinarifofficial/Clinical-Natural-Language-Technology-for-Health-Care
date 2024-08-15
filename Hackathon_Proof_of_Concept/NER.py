import spacy

# Load pre-trained model
nlp = spacy.load("en_core_sci_md")  # A model tailored for scientific/clinical text

# Sample clinical text
clinical_text = "The patient was diagnosed with diabetes and prescribed metformin."

# Process the text
doc = nlp(clinical_text)

# Extract entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
