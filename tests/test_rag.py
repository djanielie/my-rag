from rag import NO_EVIDENCE_ANSWER, RAGService
from vector_store import RetrievedChunk


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def query(self, question, top_k):
        return self.chunks[:top_k]


class FakeChatClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "The date is June 4 [project.pdf, p. 7]."}}


def test_answer_passes_source_metadata_to_grounded_prompt():
    source = RetrievedChunk(
        text="The launch date is June 4.",
        filename="project.pdf",
        page=7,
        chunk=0,
        file_hash="abc",
        distance=0.1,
    )
    client = FakeChatClient()
    service = RAGService(
        vector_store=FakeStore([source]),
        ollama_client=client,
        top_k=3,
    )

    result = service.answer("When is the launch?")

    assert result.sources == (source,)
    assert "[project.pdf, p. 7]" in result.answer
    prompt = client.calls[0]["messages"][1]["content"]
    assert 'source="project.pdf" page="7"' in prompt
    assert "When is the launch?" in prompt


def test_no_retrieved_context_declines_without_calling_model():
    client = FakeChatClient()
    service = RAGService(
        vector_store=FakeStore([]),
        ollama_client=client,
    )

    result = service.answer("What is the secret?")

    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.sources == ()
    assert client.calls == []
