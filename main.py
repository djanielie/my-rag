"""Streamlit interface for the local PDF RAG application."""

from __future__ import annotations

import hashlib
from typing import Any

import ollama
import streamlit as st

from config import CHAT_MODEL, EMBEDDING_MODEL, OLLAMA_HOST
from document_corrector import (
    DocumentCorrector,
    corrected_filename,
    export_docx,
    extract_editable_text,
)
from ingest import extract_pdf
from model_settings import (
    installed_chat_models,
    load_chat_model,
    save_chat_model,
)
from rag import RAGService
from vector_store import PDFVectorStore


st.set_page_config(page_title="Local Document AI", page_icon="📄", layout="wide")


@st.cache_resource
def create_vector_store() -> PDFVectorStore:
    return PDFVectorStore()


@st.cache_data(ttl=30)
def discover_chat_models() -> list[str]:
    return installed_chat_models(ollama.Client(host=OLLAMA_HOST))


def source_to_dict(source: Any) -> dict[str, Any]:
    return {
        "text": source.text,
        "filename": source.filename,
        "page": source.page,
        "chunk": source.chunk,
        "distance": source.distance,
    }


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)} retrieved passages)"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**{index}. {source['filename']} — page {source['page']}**"
            )
            st.text(source["text"])
            if index < len(sources):
                st.divider()


try:
    store = create_vector_store()
except Exception as exc:
    st.error(f"Could not initialize local storage: {exc}")
    st.stop()


st.title("Local Document AI")
st.caption("Question your PDFs or clean OCR text with local Ollama models.")

with st.sidebar:
    st.header("Settings")
    if st.button("Refresh installed models", use_container_width=True):
        discover_chat_models.clear()
        st.rerun()

    saved_chat_model = load_chat_model()
    try:
        chat_models = discover_chat_models()
    except Exception as exc:
        chat_models = []
        st.error(f"Could not connect to Ollama: {exc}")

    if chat_models:
        default_model = (
            saved_chat_model if saved_chat_model in chat_models else chat_models[0]
        )
        selected_chat_model = st.selectbox(
            "Answer model",
            options=chat_models,
            index=chat_models.index(default_model),
            help="The selected Ollama model generates answers from retrieved PDF text.",
        )
        if selected_chat_model != saved_chat_model:
            try:
                save_chat_model(selected_chat_model)
                st.toast(f"Answer model changed to {selected_chat_model}")
            except OSError as exc:
                st.warning(f"Could not save the model selection: {exc}")
    else:
        selected_chat_model = saved_chat_model or CHAT_MODEL
        st.warning("No installed Ollama chat models were found.")

    st.caption(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.caption(
        "The embedding model stays fixed because changing it requires "
        "re-indexing every PDF."
    )

    st.divider()
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Choose text-based PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Index selected PDFs",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        for uploaded_file in uploaded_files or []:
            try:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    extracted = extract_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )
                    added = store.add_document(extracted)

                if added:
                    st.success(
                        f"Indexed {extracted.filename}: "
                        f"{len(extracted.chunks)} chunks from "
                        f"{extracted.total_pages} pages."
                    )
                else:
                    st.info(f"{extracted.filename} is already indexed.")

                if extracted.empty_pages:
                    pages = ", ".join(map(str, extracted.empty_pages))
                    st.warning(f"No selectable text was found on page(s): {pages}.")
            except Exception as exc:
                st.error(
                    f"Could not index {uploaded_file.name}: {exc}\n\n"
                    f"Make sure Ollama is running and run "
                    f"`ollama pull {EMBEDDING_MODEL}`."
                )

    st.subheader("Indexed PDFs")
    try:
        indexed_documents = store.list_documents()
    except Exception as exc:
        st.error(f"Could not read the document index: {exc}")
        indexed_documents = []

    if not indexed_documents:
        st.caption("No PDFs indexed yet.")

    for document in indexed_documents:
        st.markdown(f"**{document.filename}**")
        st.caption(f"{document.pages} pages · {document.chunks} chunks")
        if st.button(
            "Delete",
            key=f"delete-{document.file_hash}",
            use_container_width=True,
        ):
            try:
                store.delete_document(document.file_hash)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete {document.filename}: {exc}")

rag_service = RAGService(store, chat_model=selected_chat_model)

if "messages" not in st.session_state:
    st.session_state.messages = []

rag_tab, correction_tab = st.tabs(
    ["Questions sur les PDF", "Correction de texte OCR"]
)

with rag_tab:
    messages_container = st.container(
        height=520,
        border=True,
        key="chat-history",
    )
    question = st.chat_input(
        "Posez une question sur les PDF indexés",
        disabled=not indexed_documents,
    )

    with messages_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []))

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Recherche dans les PDF et génération..."):
                        result = rag_service.answer(question)
                    sources = [source_to_dict(source) for source in result.sources]
                    st.markdown(result.answer)
                    render_sources(sources)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    error_message = (
                        f"Impossible de générer une réponse : {exc}\n\n"
                        f"Vérifiez qu’Ollama fonctionne et que "
                        f"`{selected_chat_model}` est installé."
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )

with correction_tab:
    st.subheader("Nettoyer les erreurs produites par l’OCR")
    st.info(
        "Cette fonction corrige une couche de texte OCR existante. "
        "Elle ne reconnaît pas elle-même le texte d’un PDF constitué "
        "uniquement d’images."
    )

    correction_file = st.file_uploader(
        "Importer un PDF OCRisé, DOCX, Markdown ou TXT",
        type=["pdf", "docx", "md", "markdown", "txt"],
        key="ocr-correction-file",
    )

    if correction_file:
        correction_content = correction_file.getvalue()
        correction_key = hashlib.sha256(correction_content).hexdigest()

        if len(correction_content) > 25 * 1024 * 1024:
            st.error("Le fichier dépasse la limite de 25 Mo.")
        else:
            try:
                extracted_document = extract_editable_text(
                    correction_content,
                    correction_file.name,
                )
                st.caption(
                    f"{len(extracted_document.blocks)} paragraphes/blocs · "
                    f"{len(extracted_document.text):,} caractères · "
                    f"modèle : `{selected_chat_model}`"
                )
                with st.expander("Aperçu du texte extrait"):
                    st.text_area(
                        "Texte original",
                        extracted_document.text[:20_000],
                        height=300,
                        disabled=True,
                        key=f"original-preview-{correction_key}",
                    )
                    if len(extracted_document.text) > 20_000:
                        st.caption("Aperçu limité aux 20 000 premiers caractères.")

                if st.button(
                    "Corriger le texte avec Qwen",
                    type="primary",
                    key=f"correct-{correction_key}",
                ):
                    progress_bar = st.progress(0, text="Préparation...")

                    def update_progress(current: int, total: int) -> None:
                        progress_bar.progress(
                            current / max(total, 1),
                            text=f"Correction du groupe {current}/{total}",
                        )

                    with st.spinner(
                        "Correction locale en cours. Les gros documents "
                        "peuvent prendre plusieurs minutes..."
                    ):
                        corrector = DocumentCorrector(
                            model=selected_chat_model,
                        )
                        correction_result = corrector.correct(
                            extracted_document,
                            progress=update_progress,
                        )
                        output_docx = export_docx(correction_result)

                    progress_bar.empty()
                    st.session_state.ocr_correction = {
                        "key": correction_key,
                        "model": selected_chat_model,
                        "result": correction_result,
                        "docx": output_docx,
                    }

                saved_correction = st.session_state.get("ocr_correction")
                if (
                    saved_correction
                    and saved_correction.get("key") == correction_key
                ):
                    correction_result = saved_correction["result"]
                    st.success(
                        "Correction terminée. Vérifiez le résultat avant "
                        "de remplacer un document important."
                    )
                    st.caption(
                        f"Corrigé avec `{saved_correction['model']}`. "
                        "Le fichier original n’a pas été modifié."
                    )

                    before_column, after_column = st.columns(2)
                    with before_column:
                        st.text_area(
                            "Avant",
                            correction_result.original.text[:20_000],
                            height=420,
                            disabled=True,
                            key=f"before-{correction_key}",
                        )
                    with after_column:
                        st.text_area(
                            "Après",
                            correction_result.corrected_text[:20_000],
                            height=420,
                            disabled=True,
                            key=f"after-{correction_key}",
                        )

                    if correction_result.warnings:
                        with st.expander(
                            f"Avertissements ({len(correction_result.warnings)})"
                        ):
                            for warning in correction_result.warnings:
                                st.warning(warning)

                    st.download_button(
                        "Télécharger le DOCX corrigé",
                        data=saved_correction["docx"],
                        file_name=corrected_filename(correction_file.name),
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(
                    f"Impossible de traiter {correction_file.name} : {exc}\n\n"
                    "Vérifiez le fichier et assurez-vous qu’Ollama fonctionne."
                )
