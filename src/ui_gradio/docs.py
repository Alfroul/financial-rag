"""Document management tab — upload, index, display, and delete documents."""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path

import gradio as gr

from src.config import Config
from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.loaders.pdf_loader import PDFLoader
from src.loaders.qa_loader import QALoader
from src.loaders.text_loader import TextLoader
from src.processor.chunker import TextChunker, TitleBasedChunker
from src.processor.cleaner import TextCleaner
from src.ui_gradio.services import clear_bm25, get_vectorstore

logger = logging.getLogger(__name__)
config = Config()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_TYPE_MAP = {
    ".pdf": "PDF 研报",
    ".txt": "文本",
    ".md": "Markdown",
    ".json": "Q&A (JSON)",
    ".csv": "Q&A (CSV)",
}


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _index_file(file_path: Path, original_name: str, api_key: str, chunk_size: int, chunk_overlap: int) -> tuple:
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            docs = PDFLoader().load(file_path)
        elif ext in (".txt", ".md"):
            docs = TextLoader().load(file_path)
        elif ext in (".json", ".csv"):
            docs = QALoader().load(file_path)
        else:
            return 0, f"不支持的文件格式: {ext}"
    except Exception as e:
        return 0, f"文件加载失败: {e}"

    if not docs:
        return 0, "文件内容为空"

    if config.chunker.strategy == "title":
        chunker: TextChunker | TitleBasedChunker = TitleBasedChunker(
            chunk_size=config.chunker.title_chunk_size,
            title_patterns=config.chunker.title_patterns,
        )
    else:
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    cleaner = TextCleaner()
    chunks = []
    for doc in docs:
        cleaned = cleaner.clean(doc.content)
        doc_chunks = chunker.chunk(cleaned, doc.metadata)
        chunks.extend(doc_chunks)

    if not chunks:
        return 0, "分块结果为空"

    try:
        embedder = SiliconFlowEmbedder(api_key=api_key, model=config.embedding.model)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
    except Exception as e:
        return 0, f"向量化失败: {e}"

    try:
        store = get_vectorstore()
        stem = original_name or file_path.stem
        ids = [f"{ext[1:]}_{Path(stem).stem}_{i}" for i in range(len(chunks))]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        store.add_documents(ids, documents, embeddings, metadatas)
    except Exception as e:
        return 0, f"向量存储失败: {e}"

    return len(chunks), ""


def _get_indexed_files() -> tuple[list[list[str | int]], dict[str, list[str]]]:
    """Get indexed files from vectorstore. Returns (rows, file_groups)."""
    try:
        store = get_vectorstore()
        all_ids = store.get_all_ids()
    except Exception:
        return [], {}

    file_groups: dict[str, list[str]] = {}
    for doc_id in all_ids:
        parts = doc_id.split("_", 2)
        file_key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else doc_id
        file_groups.setdefault(file_key, []).append(doc_id)

    rows: list[list[str | int]] = []
    for file_key, chunk_ids in file_groups.items():
        ext = f".{file_key.split('_')[0]}" if "_" in file_key else ""
        rows.append([file_key, _TYPE_MAP.get(ext, "未知"), len(chunk_ids)])

    return rows, file_groups


def create_docs_tab(sidebar_components: dict) -> None:
    """Create document management tab."""
    api_key_ref = sidebar_components["api_key"]
    chunk_size_ref = sidebar_components["chunk_size"]
    chunk_overlap_ref = sidebar_components["chunk_overlap"]

    gr.Markdown("### Document Management")
    gr.Markdown("Upload, index, and manage financial documents.")

    gr.Markdown("---")
    gr.Markdown("#### UPLOAD")

    file_input = gr.File(
        label="拖拽文件到此处，或点击上传",
        file_count="multiple",
        file_types=[".pdf", ".txt", ".md", ".json", ".csv"],
    )

    index_btn = gr.Button("START INDEXING", variant="primary")
    index_status = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("#### INDEXED DOCUMENTS")

    indexed_df = gr.Dataframe(
        headers=["文件名", "类型", "文档块数"],
        datatype=["str", "str", "number"],
        label="已索引文档",
        interactive=False,
        value=_get_indexed_files()[0],
    )

    refresh_btn = gr.Button("刷新列表")

    gr.Markdown("---")
    gr.Markdown("#### DELETE DOCUMENT")

    delete_file = gr.Dropdown(
        label="选择要删除的文件",
        choices=list(_get_indexed_files()[1].keys()),
        interactive=True,
    )
    delete_btn = gr.Button("删除选中文件的所有文档块")
    delete_status = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("#### RE-INDEX")
    reindex_confirm = gr.Checkbox(label="确认（不可逆）", value=False)
    reindex_btn = gr.Button("CLEAR ALL AND RE-INDEX")
    reindex_status = gr.Markdown("")

    def do_index(files, api_key, chunk_size, chunk_overlap):
        if not api_key:
            return "请先在左侧面板输入 API Key", gr.update(), gr.update()
        if not files:
            return "请选择要上传的文件", gr.update(), gr.update()

        success_count = 0
        error_msgs = []
        for f in files:
            path = Path(f.name) if hasattr(f, "name") else Path(f)
            raw_dir = _PROJECT_ROOT / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            ext = path.suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=str(raw_dir)) as tmp:
                tmp.write(path.read_bytes())
                tmp_path = Path(tmp.name)

            try:
                original_name = path.name
                chunk_count, error_msg = _index_file(
                    tmp_path, original_name, api_key, chunk_size, chunk_overlap,
                )
                if error_msg:
                    error_msgs.append(f"{original_name}: {error_msg}")
                else:
                    success_count += 1
            except Exception as e:
                error_msgs.append(f"{path.name}: {e}")
            finally:
                with contextlib.suppress(Exception):
                    tmp_path.unlink(missing_ok=True)

        if success_count > 0:
            clear_bm25()

        rows, groups = _get_indexed_files()
        status = f"Indexed {success_count}/{len(files)} files"
        if error_msgs:
            status += "\n\nFailed:\n" + "\n".join(f"- {e}" for e in error_msgs)
        return status, rows, gr.update(choices=list(groups.keys()))

    def do_refresh():
        rows, groups = _get_indexed_files()
        return rows, gr.update(choices=list(groups.keys()))

    def do_delete(selected_file, api_key):
        if not selected_file:
            return "请选择要删除的文件", gr.update(), gr.update()
        try:
            store = get_vectorstore()
            _, groups = _get_indexed_files()
            if selected_file not in groups:
                return f"文件 {selected_file} 未找到", gr.update(), gr.update()
            ids_to_delete = groups[selected_file]
            store.delete_by_ids(ids_to_delete)
            clear_bm25()
            rows, new_groups = _get_indexed_files()
            return f"已删除 {len(ids_to_delete)} 个文档块", rows, gr.update(choices=list(new_groups.keys()))
        except Exception as e:
            return f"删除失败: {e}", gr.update(), gr.update()

    def do_reindex(confirmed):
        if not confirmed:
            return "请勾选确认", gr.update(), gr.update()
        try:
            store = get_vectorstore()
            store.delete_collection()
            clear_bm25()
            rows, groups = _get_indexed_files()
            return "已清空所有文档，请重新上传", rows, gr.update(choices=list(groups.keys()))
        except Exception as e:
            return f"清空失败: {e}", gr.update(), gr.update()

    index_btn.click(
        do_index,
        [file_input, api_key_ref, chunk_size_ref, chunk_overlap_ref],
        [index_status, indexed_df, delete_file],
    )

    refresh_btn.click(do_refresh, None, [indexed_df, delete_file])

    delete_btn.click(
        do_delete,
        [delete_file, api_key_ref],
        [delete_status, indexed_df, delete_file],
    )

    reindex_btn.click(
        do_reindex,
        [reindex_confirm],
        [reindex_status, indexed_df, delete_file],
    )
