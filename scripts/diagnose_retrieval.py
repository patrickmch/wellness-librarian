#!/usr/bin/env python3
"""
Diagnostic script to analyze retrieval pipeline behavior.

Shows what happens at each stage:
1. Initial child search (embedding similarity)
2. Per-video deduplication
3. MMR diversity filtering
4. Parent expansion
5. Reranking

Usage:
    python scripts/diagnose_retrieval.py "weight loss tips"
    python scripts/diagnose_retrieval.py "magnesium for sleep" --child-top-k 50
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import numpy as np

from backend.config import get_settings
from backend.rag.providers.voyage_provider import VoyageEmbeddingProvider
from backend.rag.reranking.voyage_reranker import VoyageReranker
from backend.rag.docstore.sqlite_store import get_docstore
from backend.rag.chunking.models import ChildChunk
from backend.rag.retrieval.diversity import dedupe_by_video, maximal_marginal_relevance


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_video_distribution(results: list, get_video_id, get_title, label: str = ""):
    """Print distribution of results by video."""
    video_counts = Counter()
    video_titles = {}
    for r in results:
        vid = get_video_id(r)
        video_counts[vid] += 1
        if vid not in video_titles:
            video_titles[vid] = get_title(r)

    print(f"  Video distribution{' (' + label + ')' if label else ''}:")
    for vid, count in video_counts.most_common():
        title = video_titles[vid][:40] + "..." if len(video_titles[vid]) > 40 else video_titles[vid]
        bar = "█" * count
        print(f"    {count:2d} {bar:20s} {title}")
    print()


def diagnose_retrieval(
    query: str,
    child_top_k: int = 30,
    max_per_video: int = 2,
    mmr_lambda: float = 0.5,
    final_top_k: int = 8,
    verbose: bool = False,
):
    """Run retrieval diagnostics for a query."""
    settings = get_settings()

    print_header(f"RETRIEVAL DIAGNOSTICS")
    print(f"  Query: \"{query}\"")
    print(f"  Parameters:")
    print(f"    child_top_k:    {child_top_k}")
    print(f"    max_per_video:  {max_per_video}")
    print(f"    mmr_lambda:     {mmr_lambda}")
    print(f"    final_top_k:    {final_top_k}")

    # =========================================================================
    # STAGE 1: Embed query and search children
    # =========================================================================
    print_header("STAGE 1: Initial Child Search (Embedding Similarity)")

    print("  Embedding query with Voyage (query mode)...")
    embedding_provider = VoyageEmbeddingProvider()
    query_embedding = embedding_provider.embed_query(query)

    print(f"  Searching ChromaDB for top {child_top_k} children...")
    chroma_client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(settings.chroma_collection_v2)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=child_top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    # Process results
    child_ids = results["ids"][0]
    child_docs = results["documents"][0]
    child_metas = results["metadatas"][0]
    child_distances = results["distances"][0]
    child_embeddings = results["embeddings"][0] if results.get("embeddings") else None

    children_with_scores = []
    for i, (doc, meta, dist) in enumerate(zip(child_docs, child_metas, child_distances)):
        score = 1.0 - dist  # Convert distance to similarity
        child = ChildChunk.from_chroma_result(doc, meta)
        embedding = child_embeddings[i] if child_embeddings is not None else None
        children_with_scores.append((child, score, embedding))

    print(f"\n  Top {min(10, len(children_with_scores))} children by similarity:\n")
    print(f"  {'Rank':<5} {'Score':<7} {'Video ID':<15} {'Title':<35}")
    print(f"  {'-'*5} {'-'*7} {'-'*15} {'-'*35}")

    for i, (child, score, _) in enumerate(children_with_scores[:10]):
        title = child.title[:32] + "..." if len(child.title) > 32 else child.title
        vid_short = child.video_id[:12] + "..." if len(child.video_id) > 12 else child.video_id
        print(f"  {i+1:<5} {score:<7.4f} {vid_short:<15} {title:<35}")

    if verbose and children_with_scores:
        print(f"\n  Sample text from top result:")
        print(f"  \"{children_with_scores[0][0].text[:200]}...\"")

    print_video_distribution(
        children_with_scores,
        get_video_id=lambda x: x[0].video_id,
        get_title=lambda x: x[0].title,
        label=f"{len(children_with_scores)} children"
    )

    # =========================================================================
    # STAGE 2: Per-video deduplication
    # =========================================================================
    print_header("STAGE 2: Per-Video Deduplication")

    print(f"  Applying max {max_per_video} chunks per video...")

    after_dedup = dedupe_by_video(
        children_with_scores,
        video_id_getter=lambda x: x[0].video_id,
        max_per_video=max_per_video,
    )

    dropped_count = len(children_with_scores) - len(after_dedup)
    print(f"  Results: {len(children_with_scores)} -> {len(after_dedup)} ({dropped_count} dropped)")

    print_video_distribution(
        after_dedup,
        get_video_id=lambda x: x[0].video_id,
        get_title=lambda x: x[0].title,
        label=f"{len(after_dedup)} children"
    )

    # Show what was dropped
    if verbose:
        dropped_videos = Counter()
        kept_ids = {c[0].child_id for c in after_dedup}
        for child, score, _ in children_with_scores:
            if child.child_id not in kept_ids:
                dropped_videos[child.video_id] += 1

        if dropped_videos:
            print("  Dropped chunks by video:")
            for vid, count in dropped_videos.most_common(5):
                print(f"    {vid[:20]}: {count} chunks dropped")

    # =========================================================================
    # STAGE 3: MMR diversity filtering
    # =========================================================================
    print_header("STAGE 3: MMR Diversity Filtering")

    if child_embeddings is None:
        print("  WARNING: No embeddings returned from ChromaDB, skipping MMR")
        after_mmr = after_dedup
    else:
        print(f"  Applying MMR with lambda={mmr_lambda}...")
        print(f"  (lambda=1.0 is pure relevance, lambda=0.0 is pure diversity)")

        after_mmr = maximal_marginal_relevance(
            query_embedding=query_embedding,
            results=after_dedup,
            embedding_getter=lambda x: x[2] if x[2] is not None else query_embedding,
            lambda_param=mmr_lambda,
            k=len(after_dedup),  # Keep all, just reorder
        )

        print(f"  Results reordered by MMR score")

    print(f"\n  Top 10 after MMR reordering:\n")
    print(f"  {'Rank':<5} {'Sim':<7} {'Video ID':<15} {'Title':<35}")
    print(f"  {'-'*5} {'-'*7} {'-'*15} {'-'*35}")

    for i, (child, score, _) in enumerate(after_mmr[:10]):
        title = child.title[:32] + "..." if len(child.title) > 32 else child.title
        vid_short = child.video_id[:12] + "..." if len(child.video_id) > 12 else child.video_id
        print(f"  {i+1:<5} {score:<7.4f} {vid_short:<15} {title:<35}")

    print_video_distribution(
        after_mmr[:20],  # Top 20 after MMR
        get_video_id=lambda x: x[0].video_id,
        get_title=lambda x: x[0].title,
        label="top 20 after MMR"
    )

    # =========================================================================
    # STAGE 4: Expand to parents
    # =========================================================================
    print_header("STAGE 4: Parent Expansion")

    parent_ids = list(set(child[0].parent_id for child in after_mmr))
    print(f"  {len(after_mmr)} children map to {len(parent_ids)} unique parents")

    docstore = get_docstore()
    parents = docstore.get_batch(parent_ids)

    print(f"  Retrieved {len(parents)} parents from docstore")

    # Track which children matched which parent and best score
    parent_children_map = {}
    parent_best_score = {}
    for child, score, _ in after_mmr:
        pid = child.parent_id
        if pid not in parent_children_map:
            parent_children_map[pid] = []
            parent_best_score[pid] = 0.0
        parent_children_map[pid].append(child.child_id)
        parent_best_score[pid] = max(parent_best_score[pid], score)

    # Sort parents by best child score
    sorted_parents = sorted(
        parents.values(),
        key=lambda p: parent_best_score.get(p.parent_id, 0),
        reverse=True,
    )

    print(f"\n  Parents ranked by best child similarity:\n")
    print(f"  {'Rank':<5} {'Score':<7} {'Tokens':<7} {'Video ID':<15} {'Title':<30}")
    print(f"  {'-'*5} {'-'*7} {'-'*7} {'-'*15} {'-'*30}")

    for i, parent in enumerate(sorted_parents[:10]):
        title = parent.title[:27] + "..." if len(parent.title) > 27 else parent.title
        vid_short = parent.video_id[:12] + "..." if len(parent.video_id) > 12 else parent.video_id
        score = parent_best_score.get(parent.parent_id, 0)
        print(f"  {i+1:<5} {score:<7.4f} {parent.token_count:<7} {vid_short:<15} {title:<30}")

    print_video_distribution(
        sorted_parents,
        get_video_id=lambda x: x.video_id,
        get_title=lambda x: x.title,
        label=f"{len(sorted_parents)} parents"
    )

    # =========================================================================
    # STAGE 5: Reranking
    # =========================================================================
    print_header("STAGE 5: Voyage Reranking (Cross-Encoder)")

    if len(sorted_parents) <= 1:
        print("  Only 1 parent, skipping reranking")
        final_parents = sorted_parents[:final_top_k]
    else:
        print(f"  Reranking {len(sorted_parents)} parents with Voyage rerank-2...")
        print(f"  (Cross-encoder sees query + document together for better relevance)")

        reranker = VoyageReranker()
        parent_texts = [p.text for p in sorted_parents]

        rerank_results = reranker.rerank(
            query=query,
            documents=parent_texts,
            top_n=final_top_k,
        )

        print(f"\n  Final top {final_top_k} after reranking:\n")
        print(f"  {'Rank':<5} {'Rerank':<8} {'Orig Sim':<9} {'Video ID':<15} {'Title':<30}")
        print(f"  {'-'*5} {'-'*8} {'-'*9} {'-'*15} {'-'*30}")

        final_parents = []
        for i, r in enumerate(rerank_results):
            parent = sorted_parents[r.index]
            final_parents.append(parent)
            orig_score = parent_best_score.get(parent.parent_id, 0)
            title = parent.title[:27] + "..." if len(parent.title) > 27 else parent.title
            vid_short = parent.video_id[:12] + "..." if len(parent.video_id) > 12 else parent.video_id

            # Highlight score changes
            rerank_score = r.score
            if r.index != i:
                movement = f"(was #{r.index + 1})"
            else:
                movement = ""

            print(f"  {i+1:<5} {rerank_score:<8.4f} {orig_score:<9.4f} {vid_short:<15} {title:<25} {movement}")

    print_video_distribution(
        final_parents,
        get_video_id=lambda x: x.video_id,
        get_title=lambda x: x.title,
        label=f"final {len(final_parents)}"
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY")

    # Count videos at each stage
    stage1_videos = len(set(c[0].video_id for c in children_with_scores))
    stage2_videos = len(set(c[0].video_id for c in after_dedup))
    stage4_videos = len(set(p.video_id for p in sorted_parents))
    stage5_videos = len(set(p.video_id for p in final_parents))

    print(f"  Stage progression:")
    print(f"    1. Initial search:    {len(children_with_scores):3d} children from {stage1_videos:2d} videos")
    print(f"    2. Per-video dedup:   {len(after_dedup):3d} children from {stage2_videos:2d} videos")
    print(f"    3. MMR reordering:    {len(after_mmr):3d} children (same, reordered)")
    print(f"    4. Parent expansion:  {len(sorted_parents):3d} parents   from {stage4_videos:2d} videos")
    print(f"    5. Final reranked:    {len(final_parents):3d} parents   from {stage5_videos:2d} videos")

    # Check for dominance
    final_video_counts = Counter(p.video_id for p in final_parents)
    top_video, top_count = final_video_counts.most_common(1)[0]
    top_video_title = next(p.title for p in final_parents if p.video_id == top_video)

    print(f"\n  Most represented video in final results:")
    print(f"    \"{top_video_title}\"")
    print(f"    {top_count}/{len(final_parents)} results ({100*top_count/len(final_parents):.0f}%)")

    if top_count > len(final_parents) / 2:
        print(f"\n  ⚠️  WARNING: Single video dominates results!")
        print(f"     Consider reducing max_chunks_per_video or lowering mmr_lambda")

    # Show final context that would go to LLM
    if verbose:
        print_header("FINAL CONTEXT (would be sent to LLM)")
        for i, parent in enumerate(final_parents, 1):
            print(f"\n[{i}] From \"{parent.title}\" ({parent.category}):")
            print(f"    {parent.text[:300]}...")

    print("\n")
    return final_parents


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose retrieval pipeline behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/diagnose_retrieval.py "weight loss tips"
    python scripts/diagnose_retrieval.py "magnesium for sleep" --verbose
    python scripts/diagnose_retrieval.py "thyroid health" --max-per-video 1 --mmr-lambda 0.3
        """
    )
    parser.add_argument("query", help="Search query to diagnose")
    parser.add_argument("--child-top-k", type=int, default=30,
                        help="Number of children to search (default: 30)")
    parser.add_argument("--max-per-video", type=int, default=2,
                        help="Max chunks per video (default: 2)")
    parser.add_argument("--mmr-lambda", type=float, default=0.5,
                        help="MMR lambda: 1.0=relevance, 0.0=diversity (default: 0.5)")
    parser.add_argument("--final-top-k", type=int, default=8,
                        help="Final number of results (default: 8)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show sample text content")

    args = parser.parse_args()

    diagnose_retrieval(
        query=args.query,
        child_top_k=args.child_top_k,
        max_per_video=args.max_per_video,
        mmr_lambda=args.mmr_lambda,
        final_top_k=args.final_top_k,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
