import sys
sys.path.insert(0, ".")
from src.rag.generator import generate_grounded_answer

q1 = "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"

chunks = [
    {
        "id": "sample3_role_74",
        "chunk_id": "sample3_role_74",
        "role": "Arguments of Respondent",
        "first_para_marker": "11.8.",
        "text": (
            "This part of the impugned order has been passed in spite of bringing it to the attention of the learned Judge "
            "that there is a collateral proceeding in the NCLT seeking for the Mortgage deed to be set aside, thereby impinging "
            "on the jurisdiction of the NCLT and erred in convicting this respondent-5 and the respondents 4, 7 & 8 for wilful "
            "disobedience i.e. Civil Contempt in the absence of the prayer and based on the allegations of fraud made by the appellant."
        ),
        "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.8."}
    },
    {
        "id": "sample3_role_66",
        "chunk_id": "sample3_role_66",
        "role": "Arguments of Respondent",
        "first_para_marker": "11.3.",
        "text": (
            "The impugned order has gone on the premise that the respondent-5 was aware of the execution proceedings and "
            "the order of Attachment and Prohibitory order, while there has not been a document to evidence the same. "
            "The learned Judge has come to this conclusion solely based on the fact that the respondent-5 is the wife of the "
            "4th respondent."
        ),
        "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.3."}
    },
    {
        "id": "sample3_role_69",
        "chunk_id": "sample3_role_69",
        "role": "Arguments of Respondent",
        "first_para_marker": "11.5.",
        "text": (
            "Whereas the subject matter of the present dispute is mortgage where the transfer of ownership is conditional "
            "and not absolute as in the case of a sale or gift. The learned Judge has failed to appreciate the difference "
            "that was drawn in the Prohibitory Order aforementioned."
        ),
        "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.5."}
    }
]

res = generate_grounded_answer(question=q1, verified_chunks=chunks)
print("Answer:", res.get("answer"))
print("Citations:", res.get("citations"))
print("Status:", res.get("status"))
