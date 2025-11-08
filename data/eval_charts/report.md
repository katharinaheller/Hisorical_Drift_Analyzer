# Evaluation Analytics Report

- Files (n): **100**
- Mean NDCG@k: **0.9951** (95% CI: 0.9921 … 0.9980)
- Mean Faithfulness: **0.5649** (95% CI: 0.5509 … 0.5783)

## Korrelationsanalyse
- Pearson r: **0.157**
- Spearman ρ: **0.226**
→ Schwache bis moderate positive Abhängigkeit zwischen Ranking-Güte und Antworttreue; Faithfulness kann zusätzlich durch semantische Kohärenzmetriken (BERTScore, FactScore) kontrastiert werden.

## Method Parameters
- bootstrap iterations: `2000`
- IQR fence k: `1.5`
- z-score threshold: `3.0`

## Plots
- Histograms: `hist_*.png|svg`
- Box/Density: `box_violin_*`
- Scatter: `scatter_ndcg_vs_faithfulness.*`
- Run-order: `run_order_*.*`

## Fazit & nächste Schritte
✅ Retrieval-Architektur gilt als stabil; Optimierungspotenzial liegt im Faithfulness-Level.

🔬 **Empfohlene Experimente:**
- Vergleich mit/ohne temporale Gewichtung (`temporal_mode=True/False`) – erwarteter Δ Faithfulness ≈ 0.02–0.04.
- Dekaden-basierte Analyse (nach Implementierung der erweiterten Jahrerkennung).
- Konvergenzstudie mit n = 30, 50, 100 → Beobachtung der CI-Stabilisierung.

## Outliers (IQR or z-score flagged)

| query_id | ndcg@k | faithfulness |
|---|---:|---:|
| Compare_shallow_learning_and_deep_hierarchical_learning_structures_ | 0.9610 | 0.4608 |
| Contrast_capability_framing_versus_ethical_framing_of_AI_across_the_sources_ | 0.9150 | 0.5107 |
| Describe_expert_systems_and_their_key_components_ | 0.9866 | 0.5875 |
| Describe_how_knowledge_inference_operates_within_expert_systems_ | 0.9866 | 0.5701 |
| Describe_the_historical_transition_from_GOFAI_to_machine_learning_ | 0.9866 | 0.5360 |
| Discuss_ethical_tensions_between_automation_and_accountability_ | 0.9948 | 0.5466 |
| Discuss_how_symbolic_AI_and_connectionism_differ_conceptually_within_the_corpus_ | 1.0000 | 0.4031 |
| Explain_knowledge_representation_in_expert_systems_according_to_the_corpus_ | 0.9395 | 0.5254 |
| Explain_the_concept_of_heuristic_search_as_described_in_the_corpus_ | 1.0000 | 0.2569 |
| Explain_the_difference_between_supervised_and_unsupervised_learning_in_the_corpu | 0.9914 | 0.5402 |
| Explain_the_notion_of_machine-generated_text_threats_discussed_in_the_corpus_ | 1.0000 | 0.7388 |
| Explain_the_Turing_Test_as_described_in_the_corpus_ | 0.9948 | 0.4898 |
| List_examples_of_early_medical_expert_systems_cited_in_the_corpus_ | 0.9829 | 0.5500 |
| List_examples_of_electronics_expert_systems_cited_in_the_corpus_ | 0.9720 | 0.6335 |
| Summarize_how_context_completeness_affects_evaluation_quality_as_discussed_in_th | 0.9244 | 0.5635 |
| Summarize_how_reinforcement_learning_extended_adaptive_behavior_modeling_ | 0.9610 | 0.4632 |
| Summarize_how_search_algorithms_contributed_to_early_AI_progress_ | 0.9275 | 0.5256 |