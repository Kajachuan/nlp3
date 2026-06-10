from src.agents.orchestration.evidence_gate import LocalEvidenceGate
from src.rag.retrievers.hybrid_retriever import HybridSearchResponse, HybridSearchResult


def test_evidence_gate_uses_web_when_no_local_results() -> None:
    gate = LocalEvidenceGate(min_score=0.1, min_results=1, web_fallback_enabled=True)
    response = HybridSearchResponse(results=[], rag_hits=[], keyword_hits=[], timings_ms={})

    decision = gate.decide(response)

    assert decision.use_web_fallback
    assert decision.reason == "not enough local results"


def test_evidence_gate_respects_disabled_fallback() -> None:
    gate = LocalEvidenceGate(min_score=0.1, min_results=1, web_fallback_enabled=False)
    response = HybridSearchResponse(results=[], rag_hits=[], keyword_hits=[], timings_ms={})

    decision = gate.decide(response)

    assert not decision.use_web_fallback


def test_evidence_gate_uses_web_when_specific_product_term_is_missing() -> None:
    gate = LocalEvidenceGate(min_score=0.1, min_results=1, web_fallback_enabled=True)
    response = HybridSearchResponse(
        results=[
            HybridSearchResult(
                item={"sku": "S-BME280-I2C", "name": "Sensor ambiental BME280 I2C", "stock": 1, "price_usd": 4.25},
                score=0.45,
                semantic_score=0.45,
                keyword_score=0,
                matched_terms=["sensor"],
                sources=["rag"],
                document="Sensor ambiental BME280 I2C",
            )
        ],
        rag_hits=[],
        keyword_hits=[],
        timings_ms={},
    )

    decision = gate.decide(response, required_terms=["A1468", "sensor"])

    assert decision.use_web_fallback
    assert decision.reason == "specific product terms missing from local best result"


def test_evidence_gate_keeps_local_when_specific_product_term_matches() -> None:
    gate = LocalEvidenceGate(min_score=0.1, min_results=1, web_fallback_enabled=True)
    response = HybridSearchResponse(
        results=[
            HybridSearchResult(
                item={"sku": "S-BME280-I2C", "name": "Sensor ambiental BME280 I2C", "stock": 1, "price_usd": 4.25},
                score=0.45,
                semantic_score=0.45,
                keyword_score=0,
                matched_terms=["sensor", "bme280"],
                sources=["rag"],
                document="Sensor ambiental BME280 I2C",
            )
        ],
        rag_hits=[],
        keyword_hits=[],
        timings_ms={},
    )

    decision = gate.decide(response, required_terms=["BME280", "sensor"])

    assert not decision.use_web_fallback
