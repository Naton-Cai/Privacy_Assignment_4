import spacy

nlp = spacy.load("en_core_web_lg")

original = '''
**Subject:** Action Required: HR Update for Sarah Collins (Employee ID 0048812)

Dear Alex,

I am writing to inform you that Sarah Collins will be taking an extended medical leave of absence. Due to recent health complications, she will be staying at the Monroe Clinic for at least the next few months.

Could you please take the necessary steps to update her records in the HR system? We need to ensure her file accurately reflects her current status and contact information.

Please complete the following updates at your earliest convenience:

1. **Update Leave Status:** Change her employment status to reflect a medical leave of absence. (Employee ID: 0048812).
2. **Update Home Address:** Please ensure her recorded home address is updated to the following:
    * 77 River Road, Springfield, IL 62704

Please let me know if you require any additional documentation or information from my end to process these changes.

Thank you very much for your prompt attention to this matter.

Best regards,

David Kim
(212) 555-7890
'''

presidio = '''
**Subject:** Action Required: HR Records Update for [Employee Name]

Dear [Recipient Name],

I am writing to inform you that [Employee Name] will be taking a medical leave of absence for some time due to recent health complications.

She will be staying at [Location] for the period of [Date/Time].

To ensure her records are accurate while she is out, could you please take the following actions regarding her HR file?

1. **Update Leave Status:** Please update her HR records to reflect that she is taking medical leave.
2. **Employee ID:** Her employee ID is [Date/Time].
3. **Update Address:** Please ensure her recorded home address is updated to: [Location], [Location], IL 62704.

Please let me know if you require any additional documentation or information from my end to complete these updates.

Thank you for your prompt attention to this matter.

Best regards,

[Your Name]
[Your Cell Phone Number]
'''

llm = '''
**Subject:** Urgent HR Update: Medical Leave for <PERSON_B>

Dear <PERSON_A>,

I am writing to inform you of a necessary update regarding <PERSON_B>'s employment status. Due to recent health complications, <PERSON_B> will be taking a medical leave of absence for an indefinite period, expected to last at least a few months.

For operational and HR purposes, please note that she will be staying at <LOCATION_A> during this time.

Could you please take the following actions to update her records?

1. **Update Leave Status:** Please update her HR records to reflect that she is on medical leave.
2. **Employee ID:** Her employee ID remains <ID_NUMBER_A>.
3. **Update Address:** Please ensure her recorded home address is updated to the following: <ADDRESS_A>.

Please let me know if you require any further documentation or information from my end to process these changes. Thank you very much for handling this sensitive matter with care and efficiency.

Best regards,

<PERSON_C>
<PHONE_NUMBER_A>
'''

doc1 = nlp(original)
doc2 = nlp(presidio)
doc3 = nlp(llm)
##Similarity of two documents
similarity1 = doc1.similarity(doc2)
similarity2 = doc1.similarity(doc3)

print('output')
print(" Text vs Presidio", similarity1)
print(" Text vs Gemma4", similarity2)

original = "Draft a professional email addressed to Alex Rodriguez in regards to another employee, Sarah Collins. Inform him that Sarah is going to be MIA for a while, due to recent health complications, and that she will be staying at the Monroe Clinic for at least a few months. Let Alex know that he needs to update her HR records (her employee ID is 0048812) to reflect she is taking medical leave. Also, tell him to ensure her recorded home address is up to date. It should be 77 River Road, Springfield, IL 62704. Finally, sign the email with my name, David Kim, and my cell, (212)555-7890."
presidio = "Draft a professional email addressed to <PERSON> in regards to another employee, <PERSON>. Inform him that <PERSON> is going to be MIA for a while, due to recent health complications, and that she will be staying at <LOCATION> for <DATE_TIME>. Let <PERSON> know that he needs to update her HR records (her employee ID is <DATE_TIME>) to reflect she is taking medical leave. Also, tell him to ensure her recorded home address is up to date. It should be <LOCATION>, <LOCATION>, IL 62704. Finally, sign the email with my name, <PERSON>, and my cell, <PHONE_NUMBER>."
llm = "Draft a professional email addressed to <PERSON_A> in regards to another employee, <PERSON_B>. Inform him that <PERSON_B> is going to be MIA for a while, due to recent health complications, and that she will be staying at the <LOCATION_A> for at least a few months. Let <PERSON_A> know that he needs to update her HR records (her employee ID is <ID_NUMBER_A>) to reflect she is taking medical leave. Also, tell him to ensure her recorded home address is up to date. It should be <ADDRESS_A>. Finally, sign the email with my name, <PERSON_C>, and my cell, <PHONE_NUMBER_A>."

doc1 = nlp(original)
doc2 = nlp(presidio)
doc3 = nlp(llm)
##Similarity of two documents
similarity1 = doc1.similarity(doc2)
similarity2 = doc1.similarity(doc3)

print('input')
print(" Text vs Presidio", similarity1)
print(" Text vs Gemma4", similarity2)