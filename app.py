import argparse
import readline
import time

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy

LLM = init_chat_model(
    model='gemma4',
    model_provider='ollama',
    temperature=0
)
NLP = spacy.load('en_core_web_lg')

def presidio_anonymize(text):
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    analysis = analyzer.analyze(text=text, language='en')
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analysis,
    )
    return anonymized.text

def llm_anonymize(text):
    system_prompt = (
        'You are a local automated tool designed to strip user messages of personally identifiable information (PII). '
        'Your only function is to detect and redact PII from the input text before they reach an unprotected AI assistant. '
        'Replace all PII with sequential generic labels (e.g. <PERSON_A>, <PERSON_B>, <LOCATION_A>). '
        'Return ONLY the sanitized text without any additional commentary or added formatting.'
    )
    example_human = (
        "Call Alex Smith at (555) 123-4567. Tell Alex that his security badge "
        "(ID 9872) is ready for pickup at the Boston office."
    )
    example_ai = (
        "Call <PERSON_A> at <PHONE_NUMBER_A>. Tell <PERSON_A> that his security badge "
        "(ID <ID_NUMBER_A>) is ready for pickup at the <LOCATION_A>."
    )
    response = LLM.invoke([
        SystemMessage(system_prompt),
        HumanMessage(example_human),
        AIMessage(example_ai),
        HumanMessage(text),
    ])
    return response.content

def query_llm(prompt):
    system_prompt = (
        'You are a helpful AI assistant. '
        'The following message has been sanitized with an automated privacy tool.'
    )
    response = LLM.invoke([
        SystemMessage(system_prompt),
        HumanMessage(prompt),
    ])
    return response.content

def get_cosine(text1, text2):
    return NLP(text1).similarity(NLP(text2))

def main():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('sanitizer', help='Set to use either presidio or llm')
    
    # parse from args
    args = parser.parse_args()
    if args.sanitizer == 'presidio':
        anonymizer = presidio_anonymize
    elif args.sanitizer == 'llm':
        anonymizer = llm_anonymize
    elif args.sanitizer == 'both':
        anonymizer = lambda x: llm_anonymize(presidio_anonymize(x))
    else:
        print('Sanitizer not recognized!')
        return 1
    
    while True:
        # cmd line
        plaintext = input(f'{args.sanitizer}> ')
        if plaintext[0] == '/':
            match plaintext[1:]:
                case 'exit':
                    return
                case _:
                    print('Command not recognized!')
                    continue

        # test anonymizer
        print('Running anonymizer on example text...')
        start_time = time.perf_counter()
        cleantext = anonymizer(plaintext)
        elapsed = time.perf_counter() - start_time
        print(f'Finished after {elapsed:.2f} seconds.')
        print((
            '--- Anonymized Text ---\n'
            f'Cosine similarity: {get_cosine(plaintext, cleantext):0.5f}\n'
            f'{cleantext}\n'
        ))

        # get user input
        while (send := input(f'Send text to LLM? ').lower()[0]):
            if send in ['y', 'n']:
                break
        
        # test result on llm
        if send == 'y':
            print(f'Sending both texts to LLM...')
            resp = query_llm(cleantext)
            print((
                '--- LLM Response ---\n'
                f'{resp}\n'
            ))
   
if __name__ == '__main__':
    main()