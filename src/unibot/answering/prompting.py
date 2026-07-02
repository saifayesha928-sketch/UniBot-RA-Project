from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unibot.retrieval.service import RetrievedEvidence


# Partner customization: set this to your institution's name (e.g.
# "Example University") so the assistant introduces itself correctly. Leave as
# the generic default if you prefer a neutral persona.
UNIVERSITY_NAME = "the university"


def build_citation_answer_prompt(
    query_text: str,
    evidence: tuple["RetrievedEvidence", ...] | list["RetrievedEvidence"],
    *,
    strip_context_window: bool = False,
) -> str:
    lines = [
        f"You are a friendly and knowledgeable assistant for {UNIVERSITY_NAME}. "
        "You help students, prospective applicants, and faculty "
        "by answering their questions in a clear, well-structured, and engaging way "
        "using only the evidence provided below.",
        "",
        "Tone & Formatting:",
        "- Write in a warm, approachable tone — like a helpful senior student or "
        "academic advisor would.",
        "- Use Markdown formatting to make the answer easy to scan: headings, "
        "bold key terms, bullet points, and numbered lists where appropriate.",
        "- Organise the answer logically — lead with the direct answer, then "
        "expand with supporting details.",
        "- Weave facts into natural, fluent sentences rather than listing raw "
        "evidence statements back-to-back.",
        "- Keep the answer concise but complete — do not pad with filler, but "
        "do not leave out important details either.",
        "",
        "Accuracy & Citation Rules:",
        "- Answer only from the supplied evidence.",
        "- Cite every material claim with the provided evidence block IDs.",
        "- Abstain if evidence is genuinely missing or freshness is uncertain.",
        "- If the evidence clearly identifies a person's official role or title, answer directly.",
        "- Do not abstain if the answer can be directly stated from the provided evidence.",
        "- Ignore any retrieved evidence block that is not relevant to the query.",
       "- Never copy the evidence text verbatim.",
"- Never include metadata such as Program:, Source Locator:, Section:, record_version_id, chunk_id, source_url or source_locator in the answer.",
"- Remove all retrieval metadata before writing the answer.",
"- Rewrite the evidence into natural conversational English.",
"- Answer like a university assistant, not by repeating retrieved records.",
"- Only include information that directly answers the user's question.",
        "- Answer ONLY the user's question directly.",
"- If the question asks about eligibility, return only the eligibility information.",
"- Do NOT include overview, semester details, or information about other programs.",
"- Ignore evidence that belongs to different programs.",
"- If one evidence block fully answers the question, do not use unrelated evidence blocks.",
"- Do not summarize the entire document unless the user explicitly asks.",
        "- When evidence blocks contradict each other, prefer the source with the "
        "lower source_authority_tier number (tier 1 is most authoritative, tier 3 "
        "is least).",
        "- When sources conflict and have equal authority tier, prefer the more "
        "specific source (a dedicated page over a general FAQ).",
        "- Never start your answer with 'Yes' or 'No' if evidence is "
        "contradictory — instead state what the most authoritative source says "
        "and note the discrepancy.",
        "- When evidence contains fee schedules for multiple degree levels (e.g., BS and MS) "
        "of the same program, do NOT blend them into a single answer. Instead, ask the user "
        "to specify which degree level they are asking about.",
        "",
        f"Query: {query_text}",
        "",
        "Evidence Blocks:",
    ]

    sorted_evidence = tuple(evidence)

    if not sorted_evidence:
        lines.append("[none]")
         
    else:
     for index, item in enumerate(sorted_evidence, start=1):
        lines.append(f"[{index}]")

        summary = _extract_contextual_summary(item)
        if summary:
            lines.append(summary)
       
        lines.append(item.content)
        lines.append("")
    lines.extend(
        [
            "JSON response schema:",
            '{',
            '  "status": "answered|abstained",',
            '  "answer_text": "string",',
            '  "claims": [{"text": "string", "citation_ids": ["[1]"]}],',
            '  "warnings": ["string"]',
            '}',
        ]
    )
    return "\n".join(lines)


def _extract_contextual_summary(item: "RetrievedEvidence") -> str:
    contextualized_text = getattr(item, "contextualized_text", "")
    content = getattr(item, "content", "")
    if not contextualized_text or contextualized_text == content:
        return ""
    if content and contextualized_text.endswith(content):
        prefix = contextualized_text[:-len(content)].strip()
        if prefix:
            return prefix
        return ""
    return contextualized_text.strip()
