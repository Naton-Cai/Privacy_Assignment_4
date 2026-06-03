import argparse
import time

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

DEFAULT_TEXT = (
    'Draft a professional email addressed to Alex Rodriguez in regards to another employee, Sarah Collins. '
    'Inform him that Sarah is going to be MIA for a while, due to recent health complications, and that she will '
    'be staying at the Monroe Clinic for at least a few months. Let Alex know that he needs to update her HR '
    'records (her employee ID is 0048812) to reflect she is taking medical leave. Also, tell him to ensure her '
    'recorded home address is up to date. It should be 77 River Road, Springfield, IL 62704. '
    'Finally, sign the email with my name, David Kim, and my cell, (212)555-7890.'
)

LLM = init_chat_model(
    model='gemma4',
    model_provider='ollama',
    temperature=0
)

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

def main():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('sanitizer', help='Set to use either presidio or llm')
    parser.add_argument('--text', default=DEFAULT_TEXT, help='Prompt to use for test')
    
    # parse from args
    args = parser.parse_args()
    original_text = args.text
    if args.sanitizer == 'presidio':
        anonymizer = presidio_anonymize
    elif args.sanitizer == 'llm':
        anonymizer = llm_anonymize
    else:
        print('Sanitizer not recognized!')
        return 1
    
    # test anonymizer
    start_time = time.perf_counter()
    print(f'Running {args.sanitizer} anonymizer on example text...')
    anonymized_text = anonymizer(original_text)
    elapsed = time.perf_counter() - start_time
    print(f'Finished after {elapsed:.2f} seconds.\n')
    print((
        '--- Original Text ---\n'
        f'{original_text}\n'
        '\n'
        '--- Anonymized Text ---\n'
        f'{anonymized_text}\n'
    ))

    # test result on llm
    print(f'Sending anonymized text to LLM...')
    response = query_llm(anonymized_text)
    print((
        '--- LLM Response ---\n'
        f'{response}\n'
    ))
   
if __name__ == "__main__":
    main()