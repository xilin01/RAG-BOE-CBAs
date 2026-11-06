# RAG-BOE-CBAs

Hybrid retrieval-augmented generation (BM25 + dense embeddings + reranker) over Spanish
collective bargaining agreements published in the BOE, aimed at HR departments of small
companies. Answers cite the exact agreement, article and page.

> Work in progress. Problem statement, architecture, evaluation results and the live demo
> will be added as the project advances.

## Corpus scope and limitations

The corpus is listed in [`data/corpus.csv`](data/corpus.csv): seven state-level agreements
published in the BOE (snapshot taken in November 2026).

- Only the **original published text** of each agreement is indexed. Later amendments,
  annual salary-table updates (*actas*) and court rulings are not included, so answers may be
  outdated for articles modified after publication.
- The hospitality document (ALEH VI) is a **framework agreement**: it does not set wages,
  which are negotiated in provincial agreements.
- The engineering agreement (XX) expired on 2024-12-31 and still applies only by
  *ultraactividad* (until a new agreement replaces it).
- This is a portfolio project, not legal advice.
