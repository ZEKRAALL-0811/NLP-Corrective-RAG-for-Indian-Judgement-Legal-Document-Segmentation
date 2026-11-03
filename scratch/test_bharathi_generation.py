import sys
sys.path.insert(0, ".")
from src.rag.generator import generate_grounded_answer

q1 = "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"

chunks = [
    {
        "id": "sample3_role_68",
        "chunk_id": "sample3_role_68",
        "role": "Arguments of Respondent",
        "first_para_marker": "11.5.",
        "text": (
            "11.5. The learned Judge has failed to appreciate the interpretation of the Prohibitory order dated 21.04.2005, "
            "wherein the order prohibited the judgment debtor in alienating the property by sale, gift or otherwise. "
            "The Prohibitory order clearly states to prohibit the alienation of property by sale, gift or such transactions "
            "wherein the ownership of the property is transferred. Whereas the subject matter of the present dispute is mortgage "
            "where the transfer of ownership is conditional and not absolute as in the case of a sale or gift. "
            "The learned Judge has failed to appreciate the difference that was drawn in the Prohibitory Order aforementioned."
        ),
        "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.5."}
    },
    {
        "id": "sample3_role_70",
        "chunk_id": "sample3_role_70",
        "role": "Arguments of Respondent",
        "first_para_marker": "11.6.",
        "text": (
            "11.6. The impugned order has also failed to appreciate that the disputed Mortgage deed i.e., the Deed dated 19.06.2009 "
            "is not a new encumbrance created on the property, whereas is merely a substitution of the Mortgage deed created by "
            "Indian Bank in the year 1989. This Mortgage created by Indian Bank in 1989 was thereafter transferred to IL & FS, "
            "and then to India Bulls, and ultimately accrued in favour of this respondent-5 in lieu of the sale of Sun TV shares."
        ),
        "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.6."}
    }
]

res = generate_grounded_answer(question=q1, verified_chunks=chunks)
print("Answer:", res["answer"])
print("Citations:", res["citations"])
print("Status:", res["status"])
