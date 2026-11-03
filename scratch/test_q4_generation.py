import sys
sys.path.insert(0, ".")
from src.rag.generator import generate_grounded_answer

q4 = "How did the respondents distinguish the High Court’s powers under Article 215 of the Constitution from the procedures under the Contempt of Courts Act?"

chunks = [
    {
        "id": "sample3_role_61",
        "chunk_id": "sample3_role_61",
        "role": "Arguments of Respondent",
        "first_para_marker": "11.1.",
        "text": (
            "11.1. Mr.V.Raghavachari, the learned Senior Counsel appearing for respondent-5 would submit that "
            "when the prayer sought by the appellant is in the nature of a criminal contempt, the appellant ought to have "
            "obtained the consent of the Advocate General under Section 15 of the Contempt of Courts Act. "
            "However, the learned Judge has incorrectly applied Article 215 of the Constitution of India, which only "
            "preserves the inherent power of the High Court to punish for contempt. Article 215 ought to be read with the "
            "Contempt of Courts Act, 1971, which provides the statutory procedure and punishment."
        ),
        "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.1."}
    },
    {
        "id": "sample3_role_99",
        "chunk_id": "sample3_role_99",
        "role": "Arguments of Respondent",
        "first_para_marker": "20.",
        "text": (
            "20. The primary ground of attack on the side of the respondents/contemnors is that when the application "
            "has been filed under Article 215 of the Constitution of India, read with Section 2(b) of the Contempt of Courts Act, "
            "the procedure contemplated under the Act, 1971 has to be followed and Section 15 provides that consent of the "
            "Advocate General is mandatory for criminal contempt."
        ),
        "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "20."}
    }
]

res = generate_grounded_answer(question=q4, verified_chunks=chunks)
print("Answer:", res["answer"])
print("Citations:", res["citations"])
print("Status:", res["status"])
