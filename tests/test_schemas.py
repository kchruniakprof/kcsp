"""Tests for src/schemas — H1 SectionRecord TypedDict boundary."""


def test_section_record_importable():
    from src.schemas import SectionRecord
    assert SectionRecord is not None


def test_section_record_has_required_keys():
    from src.schemas import SectionRecord
    import typing
    hints = typing.get_type_hints(SectionRecord)
    assert set(hints.keys()) == {"section_id", "heading", "markdown", "breadcrumb"}


def test_section_record_field_types():
    from src.schemas import SectionRecord
    import typing
    hints = typing.get_type_hints(SectionRecord)
    assert hints["section_id"] is int
    assert hints["heading"] is str
    assert hints["markdown"] is str
    assert hints["breadcrumb"] is str


def test_section_record_can_be_constructed():
    from src.schemas import SectionRecord
    rec: SectionRecord = {
        "section_id": 42,
        "heading": "§1 Was ist versichert?",
        "markdown": "Der Versicherungsschutz umfasst...",
        "breadcrumb": "Kfz > Standard > §1",
    }
    assert rec["section_id"] == 42
    assert rec["markdown"] == "Der Versicherungsschutz umfasst..."


def test_generator_accepts_section_record():
    """Generator.generate() signature accepts list[SectionRecord]."""
    import inspect
    from src.generator import Generator
    sig = inspect.signature(Generator.generate)
    params = sig.parameters
    assert "sections" in params


def test_critic_evaluate_accepts_section_record():
    """Critic.evaluate() signature accepts sections parameter."""
    import inspect
    from src.critic import Critic
    sig = inspect.signature(Critic.evaluate)
    assert "sections" in sig.parameters


def test_run_critic_accepts_section_record():
    """run_critic() signature accepts sections parameter."""
    import inspect
    from src.critic import run_critic
    sig = inspect.signature(run_critic)
    assert "sections" in sig.parameters


def test_ragassistant_builds_full_markdown_for_critic(monkeypatch):
    """RAGAssistant passes r.markdown (not r.pruned_markdown) to Critic."""
    from unittest.mock import MagicMock, patch
    from src.ragassistant import RAGAssistant
    from src.query_expansion import Intent

    captured_sections = []

    def fake_run_critic(query, answer, sections, **kwargs):
        captured_sections.extend(sections)
        result = MagicMock()
        result.verdict = MagicMock()
        result.verdict.value = "PASS"
        result.answer = answer
        from src.critic import CriticVerdict
        result.verdict = CriticVerdict.PASS
        return result

    full_md = "Vollständiger Markdown-Text für den Critic."
    pruned_md = "Gekürzt."

    fake_result = MagicMock()
    fake_result.section_id = 1
    fake_result.heading = "§1"
    fake_result.markdown = full_md
    fake_result.pruned_markdown = pruned_md
    fake_result.breadcrumb = "Kfz > Standard > §1"

    fake_expanded = MagicMock()
    fake_expanded.intent = Intent.GENERAL_INFO
    fake_expanded.normalized_query = "Was ist versichert?"
    fake_expanded.paraphrases = []
    fake_expanded.section_types = []
    fake_expanded.primary_sparte = "Kfz"
    fake_expanded.sparte_hints = ["Kfz"]
    fake_expanded.domain_terms = []

    fake_qe = MagicMock()
    fake_qe.expand.return_value = fake_expanded

    fake_retriever = MagicMock()
    fake_retriever.retrieve_multi.return_value = [fake_result]

    fake_generated = MagicMock()
    fake_generated.answer = "Die Antwort."
    fake_generated.sources = [1]
    fake_generated.breadcrumbs = ["Kfz > Standard > §1"]
    fake_generator = MagicMock()
    fake_generator.generate.return_value = fake_generated

    fake_critic = MagicMock()

    rag = RAGAssistant(
        query_expansion=fake_qe,
        retriever=fake_retriever,
        generator=fake_generator,
        critic=fake_critic,
    )

    with patch("src.ragassistant.run_critic", side_effect=fake_run_critic):
        rag.ask("Was ist versichert?")

    assert len(captured_sections) > 0
    assert captured_sections[0]["markdown"] == full_md, (
        "Critic must receive full r.markdown, not r.pruned_markdown"
    )
