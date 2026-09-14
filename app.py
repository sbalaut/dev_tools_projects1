"""Two-file Streamlit multi-agent planner with memory and Langfuse traces."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import streamlit as st
from openai import OpenAI


@dataclass(frozen=True)
class FounderBrief:
    domain: str
    geography: str
    stage: str
    startup_description: str
    quality_bar: float = 8.0
    max_revisions: int = 2


@dataclass(frozen=True)
class RoleAgent:
    """A lightweight role-based agent powered by one OpenAI client."""

    role: str
    goal: str
    backstory: str
    client: OpenAI
    model: str
    temperature: float

    def execute(self, task: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                f"Role: {self.role}\nGoal: {self.goal}\n"
                f"Operating context: {self.backstory}\n"
                "Return only the requested founder-ready markdown."
            ),
            input=task,
            temperature=self.temperature,
        )
        return response.output_text


class VectorMemory:
    """Small in-session vector store used to share relevant agent outputs."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.retrieval_log: list[dict[str, Any]] = []

    def add(self, agent: str, kind: str, text: str) -> None:
        self.records.append(
            {
                "id": str(uuid.uuid4())[:8],
                "timestamp": utc_now(),
                "agent": agent,
                "kind": kind,
                "text": text,
            }
        )

    def search(self, query: str, requested_by: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.records:
            return []
        documents = [token_counts(record["text"]) for record in self.records]
        query_counts = token_counts(query)
        document_frequency = Counter(
            token for document in documents for token in set(document)
        )
        document_count = len(documents)

        def tfidf(counts: Counter[str]) -> dict[str, float]:
            return {
                token: count
                * (math.log((1 + document_count) / (1 + document_frequency[token])) + 1)
                for token, count in counts.items()
            }

        query_vector = tfidf(query_counts)
        scores = [cosine_score(query_vector, tfidf(document)) for document in documents]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        hits = [
            {
                "memory_id": self.records[index]["id"],
                "source_agent": self.records[index]["agent"],
                "kind": self.records[index]["kind"],
                "similarity": round(float(score), 4),
                "text": self.records[index]["text"],
            }
            for index, score in ranked
        ]
        self.retrieval_log.extend(
            {
                "timestamp": utc_now(),
                "requested_by": requested_by,
                "query": query,
                "memory_id": hit["memory_id"],
                "source_agent": hit["source_agent"],
                "similarity": hit["similarity"],
            }
            for hit in hits
        )
        return hits

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "id": record["id"],
                "timestamp": record["timestamp"],
                "agent": record["agent"],
                "kind": record["kind"],
                "preview": preview(record["text"], 220),
            }
            for record in self.records
        ]


def token_counts(text: str) -> Counter[str]:
    """Return lowercase word counts for the dependency-free vector memory."""

    return Counter(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def cosine_score(left: dict[str, float], right: dict[str, float]) -> float:
    common = set(left).intersection(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def preview(text: str, length: int = 180) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= length else compact[: length - 1] + "…"


def read_verdict(text: str, quality_bar: float) -> tuple[float, str, bool]:
    """Parse critic output; the numeric score is authoritative."""

    overall_matches = re.findall(
        r"^\s*(?:OVERALL\s+)?SCORE\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    score = min(10.0, max(0.0, float(overall_matches[-1]))) if overall_matches else 0.0
    verdict_matches = re.findall(
        r"^\s*VERDICT\s*:\s*(ACCEPT|REVISE)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    stated = verdict_matches[-1].upper() if verdict_matches else "REVISE"
    needs_revision = score < quality_bar
    computed = "REVISE" if needs_revision else "ACCEPT"
    return score, computed if overall_matches else stated, needs_revision


def memory_context(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "No relevant memory is available yet."
    return "\n\n".join(
        f"### Memory from {hit['source_agent']} (similarity {hit['similarity']})\n{hit['text']}"
        for hit in hits
    )


def create_agents(
    client: OpenAI, model: str, temperature: float
) -> dict[str, RoleAgent]:
    def agent(role: str, goal: str, backstory: str) -> RoleAgent:
        return RoleAgent(role, goal, backstory, client, model, temperature)

    return {
        "research": agent(
            "Research Agent",
            "Produce evidence-led market intelligence for the founder's startup domain.",
            (
                "You are a startup-accelerator market researcher. You cover market sizing, "
                "trends, customers, competitors, white space, and risk. You clearly separate "
                "verified facts, estimates, and assumptions. You do not write funding advice "
                "or pitch slides."
            ),
        ),
        "funding": agent(
            "Funding Advisor",
            "Translate research and founder stage into a realistic capital plan.",
            (
                "You advise early-stage founders on grants, accelerators, angels, funds, raise "
                "size, runway, milestones, and readiness. You rely on the Research Agent's "
                "findings and flag every programme or ticket size that needs current verification."
            ),
        ),
        "pitch": agent(
            "Pitch Coach",
            "Synthesize shared evidence into a concise investor-ready pitch-deck outline.",
            (
                "You are an experienced seed pitch coach. You create one key message per slide, "
                "cite which agent supplied each claim, keep the ask aligned with the funding "
                "plan, and never invent traction, team, market, or financial facts."
            ),
        ),
        "critic": agent(
            "Review Critic",
            "Score the pitch and provide concrete revision instructions.",
            (
                "You are a seed-fund partner. You score evidence, specificity, narrative, and "
                "investor readiness. You critique but do not rewrite the deck."
            ),
        ),
    }


def run_agent_task(
    *,
    agent: RoleAgent,
    description: str,
    expected_output: str,
    step_name: str,
    interaction: str,
    memory: VectorMemory,
    trace: list[dict[str, Any]],
    langfuse: Any,
    on_update: Callable[[str], None],
) -> str:
    on_update(f"{agent.role}: working")
    started = time.perf_counter()
    status = "completed"
    output = ""
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name=step_name,
            input={"role": agent.role, "task": description, "expected": expected_output},
            metadata={"interaction": interaction},
        ) as span:
            output = agent.execute(
                f"{description}\n\nExpected output: {expected_output}"
            )
            span.update(output={"preview": preview(output, 1500)})
    except Exception:
        status = "failed"
        raise
    finally:
        duration = round(time.perf_counter() - started, 2)
        trace.append(
            {
                "timestamp": utc_now(),
                "agent": agent.role,
                "step": step_name,
                "interaction": interaction,
                "status": status,
                "duration_seconds": duration,
            }
        )
    memory.add(agent.role, step_name, output)
    on_update(f"{agent.role}: completed in {duration:.2f}s")
    return output


def run_startup_workflow(
    *,
    brief: FounderBrief,
    openai_api_key: str,
    model: str,
    temperature: float,
    langfuse: Any,
    on_update: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run research → funding → pitch → critique → conditional revision."""

    on_update = on_update or (lambda _message: None)
    trace: list[dict[str, Any]] = []
    memory = VectorMemory()
    model_name = model.strip()
    client = OpenAI(api_key=openai_api_key)
    agents = create_agents(client, model_name, temperature)
    workflow_started = time.perf_counter()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="startup-accelerator-workflow",
        input=asdict(brief),
        metadata={"framework": "Role-based OpenAI orchestration", "model": model_name},
    ) as root_span:
        research = run_agent_task(
            agent=agents["research"],
            description=f"""Research this startup opportunity.

Domain: {brief.domain}
Geography: {brief.geography}
Startup stage: {brief.stage}
Founder description: {brief.startup_description}

Return six markdown sections:
1. TAM and SAM with the calculation basis and confidence level
2. Three to five dated market trends and implications
3. Two or three beachhead segments and their strongest pain points
4. Three named competitors with strengths and weaknesses
5. A credible white-space opportunity
6. Regulatory, execution, and market risks

Do not invent precise facts. Mark claims requiring external verification.""",
            expected_output="A structured, evidence-aware market research brief in markdown.",
            step_name="market-research",
            interaction="Founder brief → Research Agent",
            memory=memory,
            trace=trace,
            langfuse=langfuse,
            on_update=on_update,
        )

        funding_hits = memory.search(
            f"{brief.domain} market opportunity risks customers stage funding",
            requested_by="Funding Advisor",
        )
        funding = run_agent_task(
            agent=agents["funding"],
            description=f"""Prepare a funding plan for the startup below.

Founder stage: {brief.stage}
Founder description: {brief.startup_description}

Retrieved shared memory:
{memory_context(funding_hits)}

Cover:
1. Current funding-stage assessment
2. Four to six named grants, accelerators, angels, or funds and why each may fit
3. Recommended raise amount, instrument, and runway
4. Use of funds with percentages
5. Milestones the round must prove
6. Funding-readiness gaps

Flag programme availability and ticket sizes for current verification.""",
            expected_output="A six-part funding brief grounded in the Research Agent output.",
            step_name="funding-plan",
            interaction="Research Agent → Vector Memory → Funding Advisor",
            memory=memory,
            trace=trace,
            langfuse=langfuse,
            on_update=on_update,
        )

        pitch_hits = memory.search(
            "market evidence customer pain competition funding ask milestones pitch",
            requested_by="Pitch Coach",
            top_k=4,
        )
        pitch = run_agent_task(
            agent=agents["pitch"],
            description=f"""Create an investor-ready pitch-deck outline.

Founder description: {brief.startup_description}
Stage: {brief.stage}

Retrieved shared memory from prior agents:
{memory_context(pitch_hits)}

Include a startup name, a one-liner under 20 words, a two-to-three-sentence
narrative arc, and 10–12 numbered slides in standard investor order. For every
slide provide one key message, two or three bullets, and the source agent for
each factual claim. End with the exact ask recommended by the Funding Advisor.""",
            expected_output="A markdown pitch-deck outline with 10–12 slides and a precise ask.",
            step_name="initial-pitch",
            interaction="Research + Funding Memory → Pitch Coach",
            memory=memory,
            trace=trace,
            langfuse=langfuse,
            on_update=on_update,
        )

        revisions = 0
        review = ""
        score = 0.0
        verdict = "REVISE"
        previous_review = "No previous review; this is the baseline assessment."
        previous_score = 0.0
        score_history: list[dict[str, Any]] = []

        while True:
            review = run_agent_task(
                agent=agents["critic"],
                description=f"""Review this pitch outline as a seed investor.

Quality bar: {brief.quality_bar}/10
Pitch revision number: {revisions}

Pitch outline:
{pitch}

Score evidence, specificity, narrative, and investor readiness out of 10.
List concrete gaps. End with exactly:
SCORE: <overall number out of 10>
VERDICT: <ACCEPT if score is at least {brief.quality_bar}; otherwise REVISE>""",
                expected_output="Four scores, concrete gaps, SCORE line, and VERDICT line.",
                step_name=f"critic-review-{revisions + 1}",
                interaction="Pitch Coach → Review Critic",
                memory=memory,
                trace=trace,
                langfuse=langfuse,
                on_update=on_update,
            )
            score, verdict, needs_revision = read_verdict(review, brief.quality_bar)
            if not needs_revision or revisions >= brief.max_revisions:
                break

            revisions += 1
            revision_hits = memory.search(
                "pitch evidence gaps investor readiness funding alignment",
                requested_by="Pitch Coach",
                top_k=4,
            )
            pitch = run_agent_task(
                agent=agents["pitch"],
                description=f"""Revise the current pitch using the investor critic's feedback.

Current pitch:
{pitch}

Critic feedback:
{review}

Relevant shared memory:
{memory_context(revision_hits)}

Preserve correct content, address every concrete gap, keep all unverified claims
clearly labelled, and keep the final ask aligned with the Funding Advisor.""",
                expected_output="A corrected complete pitch-deck outline in markdown.",
                step_name=f"pitch-revision-{revisions}",
                interaction="Review Critic → Pitch Coach feedback loop",
                memory=memory,
                trace=trace,
                langfuse=langfuse,
                on_update=on_update,
            )

        elapsed = round(time.perf_counter() - workflow_started, 2)
        result = {
            "brief": asdict(brief),
            "research": research,
            "funding": funding,
            "pitch": pitch,
            "review": review,
            "score": score,
            "verdict": verdict,
            "revisions": revisions,
            "elapsed_seconds": elapsed,
            "trace": trace,
            "memory": memory.snapshot(),
            "retrieval_log": memory.retrieval_log,
        }
        root_span.update(
            output={
                "score": score,
                "verdict": verdict,
                "revisions": revisions,
                "elapsed_seconds": elapsed,
            }
        )

    langfuse.flush()
    return result
st.set_page_config(
    page_title="Startup Accelerator Agent Team",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, #101d35, #132847);
        border: 1px solid #27486f;
        padding: 1rem;
        border-radius: 14px;
      }
      .agent-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .7rem;
        margin: .5rem 0 1.2rem;
      }
      .agent-card {
        border: 1px solid #284566;
        background: #0f1d31;
        border-radius: 12px;
        padding: .8rem;
        color: #dce9f8;
      }
      .agent-card strong {color: #62e7c4; display: block; margin-bottom: .25rem;}
      .agent-card span {color: #9fb2ca; font-size: .84rem;}
      @media (max-width: 800px) {.agent-strip {grid-template-columns: 1fr 1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


st.title("🚀 Startup Accelerator Multi-Agent Planner")
st.caption("Research → funding strategy → pitch creation → investor feedback → revision")

st.markdown(
    """
    <div class="agent-strip">
      <div class="agent-card"><strong>Research Agent</strong><span>Market, customers, competition and risk</span></div>
      <div class="agent-card"><strong>Funding Advisor</strong><span>Capital plan, programmes and milestones</span></div>
      <div class="agent-card"><strong>Pitch Coach</strong><span>Investor narrative and slide outline</span></div>
      <div class="agent-card"><strong>Review Critic</strong><span>Scoring and revision feedback loop</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1 · Required API keys")
    st.caption("Keys stay in this browser session and are never included in downloads.")

    openai_entered = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="Configured in Secrets" if secret("OPENAI_API_KEY") else "sk-…",
        help="Required for all four role-based agents.",
    )
    langfuse_public_entered = st.text_input(
        "Langfuse public key",
        type="password",
        placeholder="Configured in Secrets" if secret("LANGFUSE_PUBLIC_KEY") else "pk-lf-…",
    )
    langfuse_secret_entered = st.text_input(
        "Langfuse secret key",
        type="password",
        placeholder="Configured in Secrets" if secret("LANGFUSE_SECRET_KEY") else "sk-lf-…",
    )
    langfuse_base_url = st.text_input(
        "Langfuse base URL",
        value=secret("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        help="Use https://us.cloud.langfuse.com for a US-region project.",
    )

    openai_api_key = openai_entered or secret("OPENAI_API_KEY")
    langfuse_public_key = langfuse_public_entered or secret("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = langfuse_secret_entered or secret("LANGFUSE_SECRET_KEY")
    credentials_ready = all(
        [openai_api_key, langfuse_public_key, langfuse_secret_key, langfuse_base_url]
    )
    if credentials_ready:
        st.success("All required credentials are available.")
    else:
        st.warning("Enter all OpenAI and Langfuse credentials to enable the workflow.")

    st.divider()
    st.header("2 · Model settings")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.25, 0.05)
    quality_bar = st.slider("Critic acceptance score", 6.0, 10.0, 8.0, 0.1)
    max_revisions = st.slider("Maximum revision passes", 1, 5, 2)

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Founder brief")
    domain = st.text_input("Startup domain", value="FinTech")
    startup_description = st.text_area(
        "Startup idea",
        height=160,
        placeholder=(
            "Example: An AI assistant for Indian MSMEs that predicts cash-flow gaps, "
            "reconciles invoices, and recommends working-capital actions."
        ),
    )

with right:
    st.subheader("Market context")
    geography = st.text_input("Target geography", value="India")
    stage = st.selectbox(
        "Startup stage",
        [
            "Idea stage",
            "Pre-seed — prototype, no revenue",
            "Pre-seed — design partners",
            "Seed — early revenue",
            "Series A — scaling",
        ],
        index=2,
    )
    st.info(
        "The Research Agent writes the evidence brief. The Funding Advisor retrieves it "
        "from vector memory. The Pitch Coach combines both outputs. The critic can return "
        "the pitch for revision until the acceptance score is reached."
    )

run_clicked = st.button(
    "Run multi-agent workflow",
    type="primary",
    use_container_width=True,
    disabled=not credentials_ready,
)

if not credentials_ready:
    st.info("Provide the four required credentials in the sidebar before starting.")

if run_clicked:
    if not domain.strip() or not startup_description.strip() or not geography.strip():
        st.warning("Complete the startup domain, idea, and geography fields.")
    else:
        try:
            from langfuse import Langfuse

            langfuse = Langfuse(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                base_url=langfuse_base_url,
            )
            with st.spinner("Validating Langfuse credentials…"):
                if not langfuse.auth_check():
                    st.error("Langfuse authentication failed. Check the keys and base URL.")
                    st.stop()

            brief = FounderBrief(
                domain=domain.strip(),
                geography=geography.strip(),
                stage=stage,
                startup_description=startup_description.strip(),
                quality_bar=quality_bar,
                max_revisions=max_revisions,
            )

            with st.status("Running the agent team…", expanded=True) as status:
                status.write("Credentials validated. Starting Research Agent.")
                result = run_startup_workflow(
                    brief=brief,
                    openai_api_key=openai_api_key,
                    model=model.strip(),
                    temperature=temperature,
                    langfuse=langfuse,
                    on_update=status.write,
                )
                status.update(label="Workflow completed", state="complete", expanded=False)

            st.session_state["workflow_result"] = result
            st.success("Founder-ready pitch outline generated successfully.")
        except Exception as exc:
            st.error(f"Workflow failed: {exc}")

result = st.session_state.get("workflow_result")
if result:
    score_col, verdict_col, revision_col, memory_col, time_col = st.columns(5)
    score_col.metric("Critic score", f"{result['score']:.1f}/10")
    verdict_col.metric("Verdict", result["verdict"])
    revision_col.metric("Revisions", result["revisions"])
    memory_col.metric("Memory items", len(result["memory"]))
    time_col.metric("Elapsed", f"{result['elapsed_seconds']:.1f}s")

    output_tab, agents_tab, trace_tab, memory_tab, architecture_tab = st.tabs(
        ["Pitch outline", "Agent contributions", "Trace & feedback", "Vector memory", "Architecture"]
    )

    with output_tab:
        st.markdown(result["pitch"])
        pitch_file = result["pitch"].encode("utf-8")
        report_file = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
        download_a, download_b = st.columns(2)
        download_a.download_button(
            "Download pitch outline (.md)",
            data=pitch_file,
            file_name="pitch_deck_outline.md",
            mime="text/markdown",
            use_container_width=True,
        )
        download_b.download_button(
            "Download complete run (.json)",
            data=report_file,
            file_name="multi_agent_run.json",
            mime="application/json",
            use_container_width=True,
        )

    with agents_tab:
        with st.expander("🔎 Research Agent", expanded=True):
            st.markdown(result["research"])
        with st.expander("💰 Funding Advisor"):
            st.markdown(result["funding"])
        with st.expander("🎤 Pitch Coach — final contribution"):
            st.markdown(result["pitch"])
        with st.expander("🧭 Review Critic"):
            st.markdown(result["review"])

    with trace_tab:
        st.subheader("Agent interaction trace")
        st.dataframe(result["trace"], use_container_width=True, hide_index=True)
        st.subheader("Final critic feedback")
        st.markdown(result["review"])
        st.caption("Detailed remote spans and timings are also sent to the configured Langfuse project.")

    with memory_tab:
        st.subheader("Shared memory contents")
        st.dataframe(result["memory"], use_container_width=True, hide_index=True)
        st.subheader("Similarity-based retrievals")
        retrieval_rows = result["retrieval_log"]
        if not retrieval_rows:
            st.info("No retrievals were recorded.")
        else:
            st.dataframe(retrieval_rows, use_container_width=True, hide_index=True)
            st.bar_chart(retrieval_rows, x="source_agent", y="similarity")

    with architecture_tab:
        st.graphviz_chart(
            """
            digraph Workflow {
              rankdir=LR;
              graph [bgcolor="transparent", pad="0.3"];
              node [shape=box, style="rounded,filled", fillcolor="#132847", color="#3a628f", fontcolor="white"];
              edge [color="#62e7c4", fontcolor="#b7c7dc"];
              Founder [shape=oval, fillcolor="#183c52"];
              Research [label="Research Agent"];
              Memory [shape=cylinder, label="Vector Memory", fillcolor="#183c52"];
              Funding [label="Funding Advisor"];
              Pitch [label="Pitch Coach"];
              Critic [label="Review Critic"];
              Output [shape=oval, label="Pitch Outline", fillcolor="#1a4b42"];
              Founder -> Research;
              Research -> Memory [label=" stores"];
              Memory -> Funding [label=" retrieves"];
              Funding -> Memory [label=" stores"];
              Memory -> Pitch [label=" retrieves"];
              Pitch -> Critic;
              Critic -> Pitch [label=" revise", color="#f3a847"];
              Critic -> Output [label=" accept"];
            }
            """,
            use_container_width=True,
        )
        st.markdown(
            """
            **Coordination logic:** the founder brief starts one controlled sequential workflow.
            Every completed agent output is stored in vector memory. Downstream agents retrieve
            the most relevant prior contributions. The Review Critic determines whether the
            pitch is accepted or sent back to the Pitch Coach, subject to the configured safety cap.
            """
        )

st.caption(
    "Training project: AI-generated market and funding claims must be verified before external or investment use."
)
