# Module 10: Root-Cause Failure Analysis (Configs B & C)

## 1. Taxonomy of Failure Causes
Every question where the generated response was a refusal, had key-fact recall < 0.5, or incurred a judged quality failure was analyzed and classified into four root causes:
1. **Category (i) - Gold Paragraph Not in Generator Context**: Retrieval failure where the gold paragraph (fair-overlap rule >= 150 characters) was omitted from the candidate chunks.
2. **Category (ii) - Router or Threshold Refusal**: The pipeline router, cross-encoder threshold, or quality gate triggered a refusal or fallback despite potential document coverage.
3. **Category (iii) - Generator Error with Correct Context**: The gold paragraph was present in context, but the LLM omitted key facts, answered incompletely, or hallucinated unsupported numbers.
4. **Category (iv) - Judge Disagreement or Noise**: The generator produced an accurate, grounded answer, but the LLM judge penalized it due to wording discrepancies (false alarm).

## 2. Failure Distribution by Category & Configuration

| Failure Category | Config B: Role RAG | Config C: Full Pipeline | Primary Root Cause Mechanism |
|---|---|---|---|
| **(i) Gold Not in Generator Context** | 4 | 3 | Role mapping filter pruned gold passage |
| **(ii) Router / Threshold Refusal** | 6 | 5 | Cross-encoder relevance scores below tau threshold |
| **(iii) Generator Error (Correct Context)** | 1 | 2 | LLM summarization brevity omitted required facts |
| **(iv) Judge Disagreement / Noise** | 0 | 1 | Conservative verdict decomposition on condensed clauses |
| **Total Failure Cases Analyzed** | **11** | **11** | Out of 22 Gold Questions |

## 3. Five Detailed Representative Failure Examples

### Example: `gold_1` (role_rag) - cat_2_router_threshold_refusal
- **Sample Document**: `sample2`
- **Final Pipeline Status**: `abstained`
- **Key-Fact Recall**: N/A (Refusal)
- **Cross-Encoder Score**: `7.45`
- **Root Cause Diagnostic**: Refusal triggered by router / threshold gate. Cross-Encoder Score: 7.45.

**Full Generated Answer**:
> The retrieved parts of this judgment do not answer this question.

<details>
<summary>Click to view full generator context</summary>

```text
--- Chunk 1 ---
(9) On 01.07.2021, the petitioners had sold/ transferred equity shares of company under Offer For Sale (O.F.S.) to the public as part of I.P.O. In the case of Pramod Swarup Agrawal, 11,11,486 equity shares were transferred for total consideration of Rs.33,00,00,000/- whereas in the case of Sneh Lata Agarwal, she sold/ transferred 14,05,405 equity shares for a total consideration of Rs.41,59,99,880/-. At the time of such transfer, the company was not a listed company. It was listed on recognized stock exchange on 05.07.2021. It is claimed that the proceeds from the sale of shares were received in the bank account of the petitioners.
--- Chunk 2 ---
They paid advance tax on the income arising out of sale of O.F.S. share in I.P.O. but before filing their income tax returns for A.Y.2022-23 after seeking consultations and opinion from various tax consultants they came to the conclusion that consideration received by them on sale/ transfer of shares to public through O.F.S. was not liable to capital gains tax under Section 45 of the Income Tax Act, 1961 (hereinafter referred to as 'the Act, 1961'). For this reason, they did not offer any tax on the said transaction in their returns filed for A.Y. 2022-23 and in fact, claimed refund of the advance tax paid. (10) According to learned counsel for the petitioner, there was no column in the income tax return which permitted petitioners to inform that the said transactions were not taxable otherwise as claimed they would have done so.
--- Chunk 3 ---
Though not very relevant but it is said that he is suffering from Alzheimer. He is a promoter shareholder of a company, namely, India Pesticides Limited. (7) Petitioner of Writ Tax No.31 of 2025 is also aged about eighty years, as claimed and a promoter shareholder of the aforesaid company. (8) Both the petitioners have been filing their income tax returns for the last more than eighteen years and it has never been the case that any notices were issued to which they did not respond

or for that matter any summons for producing any document or information or for appearance may have been issued to the said petitioners but they did not respond. They claim to be filing their returns regularly and disclosing their income.
```
</details>

### Example: `gold_2` (full_pipeline) - cat_2_router_threshold_refusal
- **Sample Document**: `sample2`
- **Final Pipeline Status**: `no_answer`
- **Key-Fact Recall**: N/A (Refusal)
- **Cross-Encoder Score**: `-3.21`
- **Root Cause Diagnostic**: Refusal triggered by router / threshold gate. Cross-Encoder Score: -3.21.

**Full Generated Answer**:
> No reliable answer found in this document for that question.

<details>
<summary>Click to view full generator context</summary>

```text
--- Chunk 1 ---
However, the petitioners through their Chartered Accountants/ Consultants filed a letter dated 16.01.2023 before the opposite party no.5 i.e. the jurisdictional assessing officer as this is the most they could do, there being no provision for uploading of such information upon the relevant portal of the Income Tax Department. Copy of the said document is annexed as Annexure no.1 in both the writ petitions. It details the reasons why petitioners were not liable to capital gain tax.
--- Chunk 2 ---
Though not very relevant but it is said that he is suffering from Alzheimer. He is a promoter shareholder of a company, namely, India Pesticides Limited. (7) Petitioner of Writ Tax No.31 of 2025 is also aged about eighty years, as claimed and a promoter shareholder of the aforesaid company. (8) Both the petitioners have been filing their income tax returns for the last more than eighteen years and it has never been the case that any notices were issued to which they did not respond

or for that matter any summons for producing any document or information or for appearance may have been issued to the said petitioners but they did not respond. They claim to be filing their returns regularly and disclosing their income.
--- Chunk 3 ---
(9) On 01.07.2021, the petitioners had sold/ transferred equity shares of company under Offer For Sale (O.F.S.) to the public as part of I.P.O. In the case of Pramod Swarup Agrawal, 11,11,486 equity shares were transferred for total consideration of Rs.33,00,00,000/- whereas in the case of Sneh Lata Agarwal, she sold/ transferred 14,05,405 equity shares for a total consideration of Rs.41,59,99,880/-. At the time of such transfer, the company was not a listed company. It was listed on recognized stock exchange on 05.07.2021. It is claimed that the proceeds from the sale of shares were received in the bank account of the petitioners.
--- Chunk 4 ---
They paid advance tax on the income arising out of sale of O.F.S. share in I.P.O. but before filing their income tax returns for A.Y.2022-23 after seeking consultations and opinion from various tax consultants they came to the conclusion that consideration received by them on sale/ transfer of shares to public through O.F.S. was not liable to capital gains tax under Section 45 of the Income Tax Act, 1961 (hereinafter referred to as 'the Act, 1961'). For this reason, they did not offer any tax on the said transaction in their returns filed for A.Y. 2022-23 and in fact, claimed refund of the advance tax paid. (10) According to learned counsel for the petitioner, there was no column in the income tax return which permitted petitioners to inform that the said transactions were not taxable otherwise as claimed they would have done so.
```
</details>

### Example: `gold_22` (full_pipeline) - cat_1_gold_not_in_context
- **Sample Document**: `sample3`
- **Final Pipeline Status**: `ready`
- **Key-Fact Recall**: `0.25`
- **Cross-Encoder Score**: `-0.87`
- **Root Cause Diagnostic**: The gold paragraph (fair-overlap >=150 chars) was not present in the generator context chunks.

**Full Generated Answer**:
> The final outcome of the original side appeals was dismissal.

<details>
<summary>Click to view full generator context</summary>

```text
--- Chunk 1 ---
27. So far as the contention raised by the respondents that the judgment debtors had paid a sum of Rs.1 Crore on 09.08.2011, whereas the amount payable as per the consent compromise decree dated 13.03.1993 was barely Rs.4 lakhs with 24 per cent interest per annum is concerned, considering the fact that no material has been placed to substantiate the said claim, we are not inclined to delve further on this. 28. For all the aforementioned reasons, all the original side appeals are dismissed. Consequently, all the connected CMPs are closed with liberty to the parties to re-apply, if required, after the NCLT decides the IBC proceedings one way or the other. Considering the facts and circumstances, there shall be no order as to costs. Needless to mention that the original title documents shall be held in custodia legis subject to further orders in the E.P.No.194 of 2004.
```
</details>

### Example: `gold_5` (full_pipeline) - cat_4_judge_noise
- **Sample Document**: `sample2`
- **Final Pipeline Status**: `ready`
- **Key-Fact Recall**: `0.50`
- **Cross-Encoder Score**: `-2.20`
- **Root Cause Diagnostic**: Judge disagreement / false alarm: Grounded factual answer penalized by judge (score: 0.67).

**Full Generated Answer**:
> The Revenue argued that the court's scope of review in this case is limited, relying on the judgment of Laljibhai Kanjibhai Mandalia [1]. They claimed there is no illegality in the impugned action as it is based on information and reason to believe. The Revenue also stated that the petitioners' counsel has proceeded with an incorrect premise by focusing solely on non-payment of capital gains tax for the sale of shares under the OFS.

<details>
<summary>Click to view full generator context</summary>

```text
--- Chunk 1 ---
(32) Learned Senior Advocate and A.S.G. appearing for the Revenue relied heavily upon the judgment of Hon'ble Supreme Court in the case of Laljibhai Kanjibhai Mandalia (supra) to submit that the scope of judicial review in the case at hand is limited and in view of the information and reason to believe referred and contained in the satisfaction note which has been produced before the Court veritably there is no scope for such review of the impugned action in this case as there is no illegality therein. Not only there is information but reason to believe and in this context he submitted that the petitioners' counsel has proceeded to argue on the incorrect premise as if non-payment of capital gains tax for the sale of shares under the OFS is the only basis for search operations under Section 132 of the Act, 1961 whereas it is not so. In this context, he asked the Court to read the satisfaction note under the heading 'Other Allegations'.
```
</details>

### Example: `gold_16` (full_pipeline) - cat_1_gold_not_in_context
- **Sample Document**: `sample3`
- **Final Pipeline Status**: `ready`
- **Key-Fact Recall**: `0.50`
- **Cross-Encoder Score**: `3.13`
- **Root Cause Diagnostic**: The gold paragraph (fair-overlap >=150 chars) was not present in the generator context chunks.

**Full Generated Answer**:
> The decree holder's counsel argued that creating a mortgage is not an alienation barred by the prohibitory order, citing Section 63 of the Insolvency & Bankruptcy Code and Section 92 of the Transfer of Property Act, 1882. [1] According to the counsel, the creation of a mortgage does not constitute an alienation within the meaning of the prohibitory order dated 21.04.2005. [2]

<details>
<summary>Click to view full generator context</summary>

```text
--- Chunk 1 ---
11.9. The learned Senior Counsel, by way of filing detailed written submissions inter alia (i) that the respondent-5 is a third party to the proceedings and not amenable to contempt jurisdiction or execution stage; (ii) that by operation of Section 92 of the Transfer of Property Act, 1882, the respondent-5 stood subrogated with effect from the date of discharge, to the rights of the original mortgagee/charge holder (Indian Bank and its assignees IL&FS and India Bulls) over the schedule property; that the finding of fraud without a trial or without adducing any evidence is untenable; that the proceeding is barred by Section 20 of the Contempt of Courts Act, 1971, as no court shall initiate any proceedings for contempt after the expiry of one year from the date on which the contempt is alleged to have been committed; that the creation of the security does not fall within the four corners of the 2005 orders; that the declaration of the loan agreement and mortgages as null and void is a jurisdictional overreach; that there is a bar of jurisdiction of this Court on account of the IBC moratorium under Section 14 and the order of NCLT dated 25.04.2022,
--- Chunk 2 ---
Moreover, Section 63 of the Insolvency & Bankruptcy Code bars the jurisdiction of the civil Court to entertain any suit or proceedings in respect of any matter seized of by the National

Company Law Tribunal. When the prohibitory order dated 21.04.2005 specifically prohibits and restrains the respondents/judgment debtors from alienating the schedule property by way of sale, gift or otherwise, it does not apply to the creation of a mortgage, which is not an alienation. So far as the execution of the money decree is concerned, it was pointed out that the judgment debtors had paid a sum of Rs.1 Crore on 09.08.2011, whereas the amount payable as per the consent compromise decree dated 13.03.1993 was barely Rs.4 lakhs with 24 per cent interest per annum.
```
</details>
