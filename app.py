from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

text = (
    'Draft a professional email addressed to Alex Rodriguez in regards to another employee, Sarah Collins. '
    'Inform him that Sarah is going to be MIA for a while, due to recent health complications, and that she will '
    'be staying at the Portland Clinic for at least a few months. Let Alex know that he needs to update her HR '
    'records (her employee ID is 0048812) to reflect she is taking medical leave. Also, tell him to ensure her '
    'recorded home address is up to date. It should be 77 River Road, Springfield, IL 62704. '
    'Finally, sign the email with my name, David Kim, and my cell, (212)555-7890.'
)

# presidio anonymization
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
analysis = analyzer.analyze(text=text, language='en')
anonymized = anonymizer.anonymize(
    text=text,
    analyzer_results=analysis,
)

# llm based anonymization
llm = init_chat_model(
    model='gemma4',
    model_provider='ollama'
)
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
response = llm.invoke([
    SystemMessage(system_prompt),
    HumanMessage(example_human),
    AIMessage(example_ai),
    HumanMessage(text),
])

print((
    '--- original text ---\n'
    f'{text}\n\n'
    '--- presidio text ---\n'
    f'{anonymized.text}\n\n'
    '--- gemma4 text ---\n'
    f'{response.content}\n\n'
))