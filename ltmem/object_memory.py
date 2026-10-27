import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ltmem.local_model_integrity import (
    assert_artifact_unchanged,
    capture_artifact_fingerprint,
    enforce_read_only_model_policy,
    verify_local_model,
)


@dataclass
class ObjectFeatures:
    object_id: str
    object_class: str
    centroid: List[float]
    volume: float
    bounding_box: Dict[str, List[float]]
    size: Dict[str, float]
    confidence: float
    session_id: str
    num_points: int
    track_id: Optional[str] = None


class ObjectMemoryRAG:
    def __init__(
        self,
        persist_dir: str,
        embedding_model: Optional[str] = None,
        embedding_model_sha256: Optional[str] = None,
        embedding_model_manifest: Optional[str] = None,
    ):
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
        from chromadb.config import Settings

        configured_model = str(embedding_model or os.getenv("BGE_MODEL_PATH", "")).strip()
        model_path, model_sha256 = verify_local_model(
            configured_model,
            embedding_model_sha256 or os.getenv("BGE_MODEL_SHA256") or "",
            embedding_model_manifest or os.getenv("BGE_MODEL_MANIFEST") or None,
            model_label="BGE model",
            path_env="BGE_MODEL_PATH",
            sha_env="BGE_MODEL_SHA256",
        )
        enforce_read_only_model_policy(
            model_path,
            specific_env="BGE_REQUIRE_READ_ONLY_MODEL",
        )
        model_fingerprint = capture_artifact_fingerprint(model_path)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self._reset_collection(
            self.chroma_client,
            "object_memory",
            metadata={
                "lt_mem_rag_schema": "1",
                "embedding_model_sha256": model_sha256,
            },
        )

        self.embed_model = HuggingFaceEmbedding(
            model_name=str(model_path),
            device="cpu",
        )
        assert_artifact_unchanged(model_path, model_fingerprint, artifact_label="BGE model")

        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context, embed_model=self.embed_model
        )

    @staticmethod
    def _reset_collection(client, collection_name: str, *, metadata: Dict[str, str]):
        from chromadb.errors import InvalidCollectionException, NotFoundError

        try:
            client.delete_collection(collection_name)
        except (NotFoundError, InvalidCollectionException):
            pass
        except ValueError as exc:
            if "does not exist" not in str(exc).lower():
                raise
        return client.create_collection(collection_name, metadata=metadata)

    def add_object(self, obj: ObjectFeatures) -> None:
        from llama_index.core import Document

        text = f"""Object: {obj.object_class}
Location: ({obj.centroid[0]:.3f}, {obj.centroid[1]:.3f}, {obj.centroid[2]:.3f}) m
Size: X={obj.size["x"]:.3f}m, Y={obj.size["y"]:.3f}m, Z={obj.size["z"]:.3f}m
Volume: {obj.volume:.6f} m³
Bounding Box: min=({obj.bounding_box["min"][0]:.3f}, {obj.bounding_box["min"][1]:.3f}, {obj.bounding_box["min"][2]:.3f}), max=({obj.bounding_box["max"][0]:.3f}, {obj.bounding_box["max"][1]:.3f}, {obj.bounding_box["max"][2]:.3f})
Confidence: {obj.confidence:.2f}
Session: {obj.session_id}"""

        metadata = {
            "object_id": obj.object_id,
            "object_class": obj.object_class,
            "centroid_x": float(obj.centroid[0]),
            "centroid_y": float(obj.centroid[1]),
            "centroid_z": float(obj.centroid[2]),
            "volume": float(obj.volume),
            "size_x": float(obj.size["x"]),
            "size_y": float(obj.size["y"]),
            "size_z": float(obj.size["z"]),
            "confidence": float(obj.confidence),
            "session_id": obj.session_id,
            "track_id": obj.track_id or "",
            "num_points": obj.num_points,
        }

        doc = Document(text=text, metadata=metadata, id_=f"{obj.object_id}_{obj.session_id}")
        existing = self.collection.get(where={"ref_doc_id": doc.id_}, include=[])
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        self.index.insert(doc)

    def search_similar_objects(
        self,
        query_text: str,
        object_class: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

        if object_class:
            query_text = f"{query_text} class:{object_class}"

        count = self.collection.count()
        if count == 0:
            return []
        retriever_kwargs: Dict[str, Any] = {"similarity_top_k": min(top_k, count)}
        if object_class:
            retriever_kwargs["filters"] = MetadataFilters(
                filters=[ExactMatchFilter(key="object_class", value=object_class)]
            )
        retriever = self.index.as_retriever(**retriever_kwargs)
        nodes = retriever.retrieve(query_text)

        results = []
        for node in nodes:
            score = node.score or 0.0
            if score >= similarity_threshold:
                results.append(
                    {
                        "object_id": node.metadata["object_id"],
                        "score": float(score),
                        "metadata": node.metadata,
                    }
                )

        return results
