import re
from happytransformer import HappyTextToText, TTSettings

grammar_model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

grammar_args = TTSettings(num_beams=5, min_length=1)

def correct_essay_grammar(essay_text, grammar_model=grammar_model, args=grammar_args):
    # Define maximum length of each chunk
    max_length = 128

    # Split essay_text into paragraphs
    paragraphs = essay_text.split('\n')

    # Define empty list to store corrected paragraphs
    corrected_paragraphs = []

    # Loop over each paragraph in paragraphs
    for paragraph in paragraphs:
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', paragraph)
        # Define empty list to store corrected sentences
        corrected_sentences = []
        # Loop over each sentence in sentences
        for sentence in sentences:
            # Split sentence into chunks of maximum length max_length
            chunks = [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]
            # Loop over each chunk in chunks
            for chunk in chunks:
                # Add the prefix "grammar: " before each input
                result = grammar_model.generate_text(f"grammar: {chunk}", args=args)
                # Append corrected chunk to corrected_sentences list
                corrected_sentences.append(result.text)
        # Concatenate corrected sentences back together
        corrected_paragraph = ' '.join(corrected_sentences)
        # Append corrected paragraph to corrected_paragraphs list
        corrected_paragraphs.append(corrected_paragraph)

    return '\n'.join(corrected_paragraphs)