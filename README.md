# Graded skill extraction over the ESCO taxonomy

This repository contains the system I submitted to the RecSys-HR 2026 WorkRB
challenge, where it finished first with a macro nDCG@100 of 0.8458. I competed
solo.

## The task

Given a piece of free text about work, the system scores every one of the 13,891
skill concepts in ESCO v1.1 and returns them in rank order. Five datasets are
scored and averaged with equal weight: four skill-extraction sets, whose queries
are job-posting sentences or segments, and one skill-normalisation set, whose
queries are short surface forms to be mapped onto a canonical entry.

Relevance is graded rather than binary. Every query and skill pair carries an
integer from 0 to 4, where 4 marks a skill the text demonstrates explicitly, 3
one it implies, 2 a taxonomic neighbour at the wrong granularity, 1 a plausible
in-domain skill, and 0 everything else. Scoring is macro-averaged nDCG@100 with
exponential gain, so the grade of an item is worth 2^grade - 1 and its position
is discounted by 1/log2(rank + 1).

Two consequences follow from that scale and they shape the whole design. Grades
1 and 2 carry real credit, so a model trained on binary targets discards it for
no gain. And only the first hundred positions are scored, which makes the
problem a cascade: recall into a candidate pool and ordering within that pool
are separable concerns with different costs.

## Method

The pipeline runs in four stages.

**Retrieval.** Twelve lanes score every concept for every query: four domain
bi-encoders, two general-purpose bi-encoders, four lexical scorers and two
string-similarity measures. The candidate pool comes from a weighted fusion of
two of them, `0.6 * z(CurriculumMatch) + 0.4 * z(ConTeXT)`, taking the top 1,000
concepts per query. Naive equal-weight reciprocal-rank fusion of the obvious
lanes performed worse than the best single lane, so the weights are fitted
rather than assumed.

**Features.** Each lane contributes a per-query z-score and the log of its
within-query rank, gathered at the candidate positions. Nine aggregate columns
follow: the base score and its rank, how many lanes place the candidate inside
their own top 50 and top 200, the mean, spread and maximum of the lane z-scores,
and the query and target word counts. That gives 33 columns per candidate.

**Reranking.** Cross-encoders score query and concept jointly, trained in two
stages. Stage one pre-trains on a subsample of a generated skill-sentence corpus
with a mined negative ladder. Stage two fine-tunes on the graded annotations
inside the candidate pool, in three query-grouped folds. All members train under
LambdaLoss with the NDCGLoss2++ scheme, which weights each pair by the nDCG
change that swapping it would produce. The submitted ensemble spans two
architecture families, DeBERTa-v3-large and Qwen3-Reranker at 4B and 8B under
LoRA.

Qwen3-Reranker ships as a causal language model that answers a yes or no
question. Loading it through a sequence-classification head attaches a randomly
initialised layer and discards what the checkpoint already knows. Instead the
scorer reads the difference between two rows of the existing output embedding,
which reduces a 151,000-way softmax to a single dot product and preserves the
pre-trained behaviour.

**Fusion.** A LightGBM ranker with the lambdarank objective combines the
retrieval features with two columns per cross-encoder member, trained with
query-grouped cross-validation on strictly out-of-fold reranker scores. Two
further blocks contribute without any GPU work. A smoothed per-skill relevance
prior, counted only over the candidate pools of each fold's training queries
with leave-one-out on training rows, supplies four columns; in the submitted
model one of them carries the highest feature importance of any column. An exact
alternative-label match is applied as a hard override at write time rather than
offered as a feature, which lifts the normalisation task and leaves the
extraction tasks unchanged.

The submission is a JSON file holding the top 500 concepts per query, keyed by
ESCO URI.

## Layout

| Path | Contents |
| --- | --- |
| `pipeline/pipeline.py` | End-to-end run: corpus and query loading, the twelve retrieval lanes, candidate export, feature construction, and the encoder cross-encoder training and scoring loop. |
| `rerankers/qwen_4b.py` | Qwen3-Reranker-4B trainer and scorer, including the converted scoring head and a ported LambdaLoss. |
| `rerankers/qwen_8b.py` | The same for Qwen3-Reranker-8B. |
| `fusion/build_submission.py` | Feature assembly, the relevance prior, the alternative-label override, the LightGBM fusion, and submission writing. |
| `model/ltr_model.py` | The benchmark `ModelInterface` implementation, so the system can be scored directly by the evaluation library. |
| `model/ltr_predict_worker.py` | Isolated prediction worker. LightGBM aborts in a process where the deep-learning framework has already been imported, so scoring runs in a separate process. |
| `model/wrb_models.py` | Local transcriptions of the benchmark's baseline retrievers, used as lanes. |

## Running it

The code depends on the WorkRB benchmark library and the challenge repository
published by the organisers, on the ESCO v1.1 skill list, and on the model
checkpoints named in the source. Neither the challenge data nor any trained
weights are included here; the data is the organisers' to distribute, and the
weights are large and reproducible from the code.

Training ran on a Slurm cluster across A100-80GB, H100-96GB and H200-141GB
nodes, for a campaign total of roughly 250 to 300 single-GPU hours. Both
trainers size their step count from measured throughput against a wall-clock
budget rather than a fixed epoch count, so the learning-rate schedule still
decays to zero under a hard deadline, and both catch out-of-memory errors to
halve the batch and double the step count so the same number of rows is seen
either way.

The job scripts that launched these runs are specific to one cluster and are not
included.

## Notes

Diagnostic and ablation scripts written during development are omitted. What is
here is the path that produced the submitted system.

Stage-one training data is `TechWolf/Synthetic-ESCO-skill-sentences`, released
under CC-BY-4.0. The target taxonomy is ESCO v1.1. Model checkpoints are the
publicly released weights named in the source files and remain under their own
licences.
