import json
from pathlib import Path

# 70 diverse questions
questions_data = [
    # Facts (8)
    {"id": "facts_01", "gold_role": "Facts", "gold_kind": "role", "question": "What transactions led to the dispute between the builder and the allottees?"},
    {"id": "facts_02", "gold_role": "Facts", "gold_kind": "role", "question": "How much consideration was paid under the initial agreement for sale in 1987?"},
    {"id": "facts_03", "gold_role": "Facts", "gold_kind": "role", "question": "When was the search operation initiated under Section 132 against the company?"},
    {"id": "facts_04", "gold_role": "Facts", "gold_kind": "role", "question": "What were the findings of the assessing officer during the scrutiny assessment?"},
    {"id": "facts_05", "gold_role": "Facts", "gold_kind": "role", "question": "What events transpired before the trial court prior to the filing of the first appeal?"},
    {"id": "facts_06", "gold_role": "Facts", "gold_kind": "role", "question": "Did the tenant deposit the arrears of rent before the rent controller?"},
    {"id": "facts_07", "gold_role": "Facts", "gold_kind": "role", "question": "What was the background of the compromise decree entered into by the partners?"},
    {"id": "facts_08", "gold_role": "Facts", "gold_kind": "role", "question": "Which bank issued the letter of credit and on what date was it dishonoured?"},

    # Issue (8)
    {"id": "issue_01", "gold_role": "Issue", "gold_kind": "role", "question": "What was the central question of law formulated by the division bench?"},
    {"id": "issue_02", "gold_role": "Issue", "gold_kind": "role", "question": "Did the High Court frame an issue regarding the retrospective effect of the statutory amendment?"},
    {"id": "issue_03", "gold_role": "Issue", "gold_kind": "role", "question": "What point for consideration was referred to the larger constitutional bench?"},
    {"id": "issue_04", "gold_role": "Issue", "gold_kind": "role", "question": "Is the writ petition maintainable against an interlocutory order passed under Section 148?"},
    {"id": "issue_05", "gold_role": "Issue", "gold_kind": "role", "question": "What was the primary issue regarding the jurisdiction of the National Company Law Tribunal?"},
    {"id": "issue_06", "gold_role": "Issue", "gold_kind": "role", "question": "Does the bar under Section 139(8A) operate when search proceedings have already commenced?"},
    {"id": "issue_07", "gold_role": "Issue", "gold_kind": "role", "question": "Did the court frame any question concerning the violation of natural justice principles?"},
    {"id": "issue_08", "gold_role": "Issue", "gold_kind": "role", "question": "What legal point was taken up for adjudication regarding the validity of the warrant?"},

    # Arguments of Petitioner (8)
    {"id": "pet_01", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "What did learned Senior Counsel for the appellant contend regarding the validity of the mortgage?"},
    {"id": "pet_02", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "How did the petitioner challenge the satisfaction note recorded by the director general?"},
    {"id": "pet_03", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "What submissions were advanced by the appellant on the interpretation of Section 55(6)?"},
    {"id": "pet_04", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "Why did the petitioner argue that the reopening of assessment was time-barred?"},
    {"id": "pet_05", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "What was the petitioner's argument on the non-applicability of the insolvency moratorium?"},
    {"id": "pet_06", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "Did the appellant contend that fraud vitiates all subsequent execution proceedings?"},
    {"id": "pet_07", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "What plea did the petitioner urge regarding the absence of reason to believe?"},
    {"id": "pet_08", "gold_role": "Arguments of Petitioner", "gold_kind": "role", "question": "How did the appellant's counsel explain the delay in approaching the appellate authority?"},

    # Arguments of Respondent (8)
    {"id": "resp_01", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "What did the Additional Solicitor General submit in defense of the revenue's action?"},
    {"id": "resp_02", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "How did the respondent justify the invocation of Section 131(1A) powers?"},
    {"id": "resp_03", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "What was the respondent's argument regarding the exclusive jurisdiction of the NCLT?"},
    {"id": "resp_04", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "Why did counsel for the opposite party contend that the writ petition was premature?"},
    {"id": "resp_05", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "What counter-submission was made by the respondent regarding subrogation of the earlier mortgage?"},
    {"id": "resp_06", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "How did the department defend the authorization order under Section 132(1)?"},
    {"id": "resp_07", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "What objections did the respondents raise concerning the maintainability of contempt proceedings?"},
    {"id": "resp_08", "gold_role": "Arguments of Respondent", "gold_kind": "role", "question": "Did the respondent argue that the consent decree had already been superseded by subsequent orders?"},

    # Reasoning (8)
    {"id": "reas_01", "gold_role": "Reasoning", "gold_kind": "role", "question": "On what reasoning did the court conclude that the satisfaction note was defective?"},
    {"id": "reas_02", "gold_role": "Reasoning", "gold_kind": "role", "question": "Why did the bench hold that post-search material cannot validate an illegal authorization?"},
    {"id": "reas_03", "gold_role": "Reasoning", "gold_kind": "role", "question": "What legal rationale did the court apply to distinguish the judgment of the Supreme Court?"},
    {"id": "reas_04", "gold_role": "Reasoning", "gold_kind": "role", "question": "How did the High Court interpret the statutory phrase 'reason to believe' in Section 132?"},
    {"id": "reas_05", "gold_role": "Reasoning", "gold_kind": "role", "question": "Why did the judges hold that inherent powers under Article 215 remain untrammeled?"},
    {"id": "reas_06", "gold_role": "Reasoning", "gold_kind": "role", "question": "On what grounds did the court reject the department's reliance on the supplementary affidavit?"},
    {"id": "reas_07", "gold_role": "Reasoning", "gold_kind": "role", "question": "Why did the bench find that creating a mortgage amounts to an alienation of property?"},
    {"id": "reas_08", "gold_role": "Reasoning", "gold_kind": "role", "question": "How did the court balance the rights of the decree holder against the IBC moratorium?"},

    # Decision (8)
    {"id": "dec_01", "gold_role": "Decision", "gold_kind": "role", "question": "What operative directions were issued by the High Court in the concluding paragraph?"},
    {"id": "dec_02", "gold_role": "Decision", "gold_kind": "role", "question": "Was the impugned warrant of authorization quashed by the bench?"},
    {"id": "dec_03", "gold_role": "Decision", "gold_kind": "role", "question": "What final order was passed regarding the pending execution petitions?"},
    {"id": "dec_04", "gold_role": "Decision", "gold_kind": "role", "question": "Did the court grant costs or order the parties to bear their own expenses?"},
    {"id": "dec_05", "gold_role": "Decision", "gold_kind": "role", "question": "How were the original side appeals finally disposed of?"},
    {"id": "dec_06", "gold_role": "Decision", "gold_kind": "role", "question": "What relief was given to the petitioners regarding the impounded title documents?"},
    {"id": "dec_07", "gold_role": "Decision", "gold_kind": "role", "question": "Was the notice issued under Section 131(1A) set aside by the court?"},
    {"id": "dec_08", "gold_role": "Decision", "gold_kind": "role", "question": "Did the appellate court confirm or reverse the order passed by the single judge?"},

    # None (8)
    {"id": "none_01", "gold_role": "None", "gold_kind": "role", "question": "Which judges comprised the division bench that heard this matter?"},
    {"id": "none_02", "gold_role": "None", "gold_kind": "role", "question": "On which date was the judgment reserved and when was it pronounced?"},
    {"id": "none_03", "gold_role": "None", "gold_kind": "role", "question": "What was the neutral citation number assigned to this judgment?"},
    {"id": "none_04", "gold_role": "None", "gold_kind": "role", "question": "Who appeared as counsel for the income tax department?"},
    {"id": "none_05", "gold_role": "None", "gold_kind": "role", "question": "In which high court was this writ petition decided?"},
    {"id": "none_06", "gold_role": "None", "gold_kind": "role", "question": "What are the names and addresses of the contesting respondents?"},
    {"id": "none_07", "gold_role": "None", "gold_kind": "role", "question": "What was the original side appeal number of the lead appeal?"},
    {"id": "none_08", "gold_role": "None", "gold_kind": "role", "question": "Under which court roster was this matter listed for hearing?"},

    # Out of Scope (8)
    {"id": "oos_01", "gold_role": None, "gold_kind": "out_of_scope", "question": "Can you recommend a good Italian restaurant in Bangalore?"},
    {"id": "oos_02", "gold_role": None, "gold_kind": "out_of_scope", "question": "What is the current exchange rate between US Dollar and Indian Rupee?"},
    {"id": "oos_03", "gold_role": None, "gold_kind": "out_of_scope", "question": "Write a JavaScript script to fetch data from an external REST API."},
    {"id": "oos_04", "gold_role": None, "gold_kind": "out_of_scope", "question": "Who won the best actor award at the National Film Awards last year?"},
    {"id": "oos_05", "gold_role": None, "gold_kind": "out_of_scope", "question": "What are the best places to visit in Himachal Pradesh during winter?"},
    {"id": "oos_06", "gold_role": None, "gold_kind": "out_of_scope", "question": "How do I bake sourdough bread with whole wheat flour?"},
    {"id": "oos_07", "gold_role": None, "gold_kind": "out_of_scope", "question": "Can you tell me a humorous joke to brighten my morning?"},
    {"id": "oos_08", "gold_role": None, "gold_kind": "out_of_scope", "question": "What are the health benefits of green tea compared to black coffee?"},

    # Whole Document (6)
    {"id": "doc_01", "gold_role": None, "gold_kind": "whole_document", "question": "Please provide a comprehensive summary of this entire judgment."},
    {"id": "doc_02", "gold_role": None, "gold_kind": "whole_document", "question": "Can you give me an executive brief of the facts, arguments, and final ruling?"},
    {"id": "doc_03", "gold_role": None, "gold_kind": "whole_document", "question": "Explain the gist and overall outcome of this legal case in plain language."},
    {"id": "doc_04", "gold_role": None, "gold_kind": "whole_document", "question": "What is this legal dispute about from start to finish?"},
    {"id": "doc_05", "gold_role": None, "gold_kind": "whole_document", "question": "Summarize the key takeaways and major holdings of this decision."},
    {"id": "doc_06", "gold_role": None, "gold_kind": "whole_document", "question": "Give me a complete walkthrough of this court judgment."},
]

# Split 50/50 stratified into dev and test
dev_set = []
test_set = []

# Group by category
categories = {}
for q in questions_data:
    cat = q["gold_kind"] if q["gold_kind"] in ["out_of_scope", "whole_document"] else q["gold_role"]
    categories.setdefault(cat, []).append(q)

for cat, items in categories.items():
    n = len(items)
    n_dev = n // 2
    for i, item in enumerate(items):
        item_copy = dict(item)
        if i < n_dev:
            item_copy["split"] = "dev"
            dev_set.append(item_copy)
        else:
            item_copy["split"] = "test"
            test_set.append(item_copy)

benchmark_data = {
    "total_questions": len(questions_data),
    "dev_count": len(dev_set),
    "test_count": len(test_set),
    "dev": dev_set,
    "test": test_set,
}

out_file = Path("artifacts/role_mapper_benchmark.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2)

print(f"Saved benchmark to {out_file}: total={len(questions_data)}, dev={len(dev_set)}, test={len(test_set)}")
