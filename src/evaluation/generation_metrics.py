"""Generation metrics for RAG evaluation.

Three metrics scored per answer:

    * Faithfulness       (0..1) — LLM-as-judge: are the answer's claims
                                  supported by the retrieved context?
    * Answer relevance   (0..1) — LLM-as-judge: does the answer actually
                                  address the question?
    * Refusal correctness (0/1) — pure rule: for should_refuse=True
                                  questions, did the system refuse?

Design notes:
  * The judge is our Ollama LLM. Same model, same quality trade-offs.
    When we report numbers, we note the judge model so an interviewer
    can read the scores in context.
  * Each judge call parses a strict JSON response. We clamp scores to
    [0, 1] and default to 0 on parse failure — safer than defaulting
    to 1, which would hide bugs.
  * Judge temperature is 0.0 — we want deterministic grading.
  * We never pass hallucinated citations to the judge. The faithfulness
    check uses the answer AFTER the hallucinated-citation strip (what
    the user actually sees).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import mean
from typing import Optional

from loguru import logger

from src.evaluation.benchmark import BenchmarkQuestion
from src.evaluation.retrieval_metrics import QuestionResult
from src.generation.llm_client import generate
from src.generation.rag import GroundedAnswer
from src.retrieval.vector_search import RetrievedChunk


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = """\
You are an evaluation judge. Your job: decide whether an ANSWER's claims \
are supported by the provided CONTEXT. You never consider outside \
knowledge — only the CONTEXT. Reply with VALID JSON ONLY, no markdown, \
no commentary.
"""

_FAITHFULNESS_USER = """\
QUESTION:
{question}

CONTEXT (the retrieved sources the answer was generated from):
{context}

ANSWER (to be judged):
{answer}

Task: list the distinct factual CLAIMS the ANSWER makes. For each claim, \
decide whether it is supported by the CONTEXT alone (ignore your own \
knowledge). A claim is supported if the CONTEXT states or clearly \
implies it. Boilerplate like "the sources describe X" is not a claim.

Reply with VALID JSON ONLY in this exact shape:
{{
  "claims": [
    {{"text": "<claim 1>", "supported": true}},
    {{"text": "<claim 2>", "supported": false}}
  ]
}}
"""


_RELEVANCE_SYSTEM = """\
You are an evaluation judge. Your job: rate how directly an ANSWER \
addresses a QUESTION on a 0.0–1.0 scale. 1.0 means the ANSWER directly \
and completely addresses the QUESTION. 0.0 means the ANSWER is off-topic \
or evasive. Reply with VALID JSON ONLY, no markdown, no commentary.
"""

_RELEVANCE_USER = """\
QUESTION:
{question}

ANSWER:
{answer}

Task: score how well the ANSWER addresses the QUESTION. Ignore factual \
correctness (that's judged separately) — just rate how on-topic the \
answer is.

Reply with VALID JSON ONLY in this exact shape:
{{
  "relevance": 0.85,
  "reason": "<one short sentence>"
}}
"""


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a response. Returns None on failure.

    Small models sometimes wrap JSON in markdown fences or add commentary.
    We find the first {...} block greedily, which handles both.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Judge calls
# ---------------------------------------------------------------------------

def judge_faithfulness(
    question: str,
    answer: str,
    context_chunks: list[RetrievedChunk],
) -> float:
    """Return faithfulness score 0..1 = supported_claims / total_claims.

    Returns 1.0 if the answer has no identifiable claims (vacuous but
    not unfaithful). Returns 0.0 on judge parse failure (fail loud).
    """
    if not context_chunks:
        return 0.0

    context = "\n\n".join(
        f"[{c.metadata.cve_id or c.document_id}]: {c.text}"
        for c in context_chunks
    )

    user = _FAITHFULNESS_USER.format(
        question=question, context=context, answer=answer,
    )
    try:
        resp = generate(
            system=_FAITHFULNESS_SYSTEM,
            user=user,
            temperature=0.0,
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning(f"[eval] faithfulness judge call failed: {e}")
        return 0.0

    parsed = _parse_json(resp.text)
    if parsed is None or "claims" not in parsed:
        logger.warning(f"[eval] faithfulness judge returned unparseable JSON: {resp.text[:200]!r}")
        return 0.0

    claims = parsed["claims"]
    if not isinstance(claims, list) or not claims:
        return 1.0  # no claims → vacuously faithful

    supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported"))
    return supported / len(claims)


def judge_relevance(question: str, answer: str) -> float:
    """Return 0..1 relevance score. 0.0 on judge failure."""
    user = _RELEVANCE_USER.format(question=question, answer=answer)
    try:
        resp = generate(
            system=_RELEVANCE_SYSTEM,
            user=user,
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as e:
        logger.warning(f"[eval] relevance judge call failed: {e}")
        return 0.0

    parsed = _parse_json(resp.text)
    if parsed is None or "relevance" not in parsed:
        logger.warning(f"[eval] relevance judge returned unparseable JSON: {resp.text[:200]!r}")
        return 0.0

    try:
        score = float(parsed["relevance"])
    except (TypeError, ValueError):
        return 0.0

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Per-question generation scoring
# ---------------------------------------------------------------------------

@dataclass
class GenerationScore:
    """Generation-side scores for one answered question."""
    question_id: str
    question_type: str
    faithfulness: Optional[float]        # None for refusals (no claims to judge)
    relevance: Optional[float]           # None for refusals
    refusal_correct: Optional[bool]      # None for non-refusal questions
    answered: bool                       # True if the system produced an answer

    def summary(self) -> str:
        parts = [f"{self.question_id}"]
        if self.faithfulness is not None:
            parts.append(f"faith={self.faithfulness:.2f}")
        if self.relevance is not None:
            parts.append(f"rel={self.relevance:.2f}")
        if self.refusal_correct is not None:
            parts.append(f"refusal={'✓' if self.refusal_correct else '✗'}")
        return "  ".join(parts)


def score_generation(
    q: BenchmarkQuestion,
    ga: GroundedAnswer,
) -> GenerationScore:
    """Compute generation metrics for one (question, grounded-answer) pair.

    Routing:
      * `should_refuse=True`     → only check refusal_correct.
      * refused when it shouldn't → faithfulness/relevance = 0.0.
      * normal answer            → both judges run.
    """
    answered = not ga.is_insufficient_evidence

    # Case 1: unanswerable negative test
    if q.should_refuse:
        return GenerationScore(
            question_id=q.id,
            question_type=q.question_type,
            faithfulness=None,
            relevance=None,
            refusal_correct=ga.is_insufficient_evidence,
            answered=answered,
        )

    # Case 2: answerable question, but system refused — score 0 on both.
    if ga.is_insufficient_evidence:
        return GenerationScore(
            question_id=q.id,
            question_type=q.question_type,
            faithfulness=0.0,
            relevance=0.0,
            refusal_correct=None,
            answered=False,
        )

    # Case 3: answered question — run both judges.
    faith = judge_faithfulness(q.question, ga.answer, ga.retrieved)
    rel = judge_relevance(q.question, ga.answer)
    return GenerationScore(
        question_id=q.id,
        question_type=q.question_type,
        faithfulness=faith,
        relevance=rel,
        refusal_correct=None,
        answered=True,
    )


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

@dataclass
class GenerationReport:
    """Aggregated generation-metric report for one retriever + benchmark run."""
    retriever_name: str
    n_answered: int
    n_refused_correctly: int
    n_refused_incorrectly: int
    n_should_have_refused: int
    mean_faithfulness: float
    mean_relevance: float
    refusal_accuracy: float   # correct refusals / total should_refuse
    per_question: list[GenerationScore]

    def summary(self) -> str:
        return (
            f"[{self.retriever_name:20s}] "
            f"answered={self.n_answered:2d}  "
            f"faith={self.mean_faithfulness:.3f}  "
            f"rel={self.mean_relevance:.3f}  "
            f"refusal_acc={self.refusal_accuracy:.2f} "
            f"({self.n_refused_correctly}/{self.n_should_have_refused})"
        )


def aggregate_generation(
    retriever_name: str,
    scores: list[GenerationScore],
) -> GenerationReport:
    """Average generation scores into a GenerationReport."""
    answered = [s for s in scores if s.answered and s.faithfulness is not None]
    refusal_cases = [s for s in scores if s.refusal_correct is not None]

    n_should_refuse = len(refusal_cases)
    n_refused_correctly = sum(1 for s in refusal_cases if s.refusal_correct)
    n_refused_incorrectly = n_should_refuse - n_refused_correctly

    mean_faith = mean(s.faithfulness for s in answered) if answered else 0.0
    mean_rel = mean(s.relevance for s in answered) if answered else 0.0
    refusal_acc = (
        n_refused_correctly / n_should_refuse if n_should_refuse else 0.0
    )

    return GenerationReport(
        retriever_name=retriever_name,
        n_answered=len(answered),
        n_refused_correctly=n_refused_correctly,
        n_refused_incorrectly=n_refused_incorrectly,
        n_should_have_refused=n_should_refuse,
        mean_faithfulness=mean_faith,
        mean_relevance=mean_rel,
        refusal_accuracy=refusal_acc,
        per_question=scores,
    )


# ---------------------------------------------------------------------------
# CLI smoke test — judges on one real query
# ---------------------------------------------------------------------------

def main() -> None:
    """Run both judges once against a real Log4Shell answer."""
    from src.generation.rag import BaselineRAG

    rag = BaselineRAG.with_hybrid_rerank()
    q = BenchmarkQuestion(
        id="SMOKE",
        question="What is Log4Shell?",
        question_type="factual",
        expected_cve_ids=["CVE-2021-44228"],
    )

    print(f"Running RAG on: {q.question!r}")
    ga = rag.answer(q.question)
    print(f"Got answer ({len(ga.answer)} chars), {len(ga.retrieved)} chunks")
    print()

    print("Judging...")
    score = score_generation(q, ga)
    print(score.summary())
    print()
    print("Full answer:")
    print(ga.answer)


if __name__ == "__main__":
    main()