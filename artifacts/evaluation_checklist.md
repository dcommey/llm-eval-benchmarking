# LLM Evaluation Checklist

A practical checklist for evaluating LLM-powered applications. Derived from the survey "Evaluating LLM Applications: Benchmarks, Metrics, Test Sets, and Monitoring Pipelines."

---

## Minimum Viable Evaluation Suite

Before deploying any LLM application:

- [ ] **Defined quality dimensions.** Document which dimensions (correctness, helpfulness, harmlessness, groundedness, format adherence) matter most.
- [ ] **Golden test set.** At least 50–100 curated examples with reference answers covering core functionality.
- [ ] **Edge case coverage.** Adversarial prompts, out-of-scope queries, boundary conditions.
- [ ] **Automated assertions.** Format validation, required/excluded content checks.
- [ ] **At least one semantic metric.** BERTScore, embedding similarity, or equivalent.
- [ ] **Human evaluation baseline.** 50+ examples scored by humans to calibrate automated metrics.
- [ ] **Prompt/model version control.** All prompts versioned with documented changes.

---

## Pre-Deployment Checklist

Before releasing a new version:

- [ ] Run complete regression test suite
- [ ] Compare metrics against previous version; investigate regressions
- [ ] Review random sample of outputs manually
- [ ] Test adversarial and edge cases
- [ ] Verify format compliance on all test examples
- [ ] For RAG: check retrieval quality and citation accuracy
- [ ] Document known limitations and failure cases
- [ ] Prepare rollback plan with previous version ready

---

## Production Monitoring Checklist

For ongoing production systems:

- [ ] **Latency monitoring.** Track p50, p95, p99 response times
- [ ] **Error rate tracking.** Parse failures, API errors, timeouts
- [ ] **User feedback collection.** Thumbs up/down, explicit ratings, escalations
- [ ] **Output sampling.** Regular review of random production outputs
- [ ] **Automated quality scoring.** Run LLM-as-judge or metric computation on sampled outputs
- [ ] **Alerting.** Automated alerts for metric degradation or error spikes
- [ ] **Canary tests.** Periodic synthetic queries to detect behavioral changes
- [ ] **Model update protocol.** Process for testing after LLM provider updates

---

## RAG-Specific Checklist

For retrieval-augmented generation systems:

- [ ] Separate retrieval and generation evaluation
- [ ] Measure retrieval Recall@k and Precision@k
- [ ] Evaluate faithfulness to retrieved documents
- [ ] Check for "correct but unsupported" responses
- [ ] Verify citation accuracy and coverage
- [ ] Test out-of-scope queries (information not in knowledge base)
- [ ] Monitor retrieval latency and index freshness

---

## Human Evaluation Checklist

When conducting human evaluations:

- [ ] Define clear rubrics with anchor examples
- [ ] Train evaluators with calibration exercises
- [ ] Measure inter-rater reliability (Kappa > 0.6)
- [ ] Randomize presentation order for pairwise comparisons
- [ ] Blind evaluators to model identity where possible
- [ ] Include attention checks to ensure evaluator engagement
- [ ] Document and resolve disagreements
- [ ] Report sample sizes and confidence intervals

---

## LLM-as-Judge Checklist

When using LLM evaluators:

- [ ] Use a different model than the one being evaluated
- [ ] Provide explicit rubrics in the evaluation prompt
- [ ] Request chain-of-thought reasoning before scores
- [ ] Randomize presentation order for comparisons
- [ ] Validate scores against human judgments on a sample
- [ ] Use multiple judge models where feasible
- [ ] Document known biases in your report

---

## Quality Dimensions Reference

| Dimension | Definition |
|-----------|------------|
| **Correctness** | Factually accurate, logically sound, no errors |
| **Helpfulness** | Addresses user intent, actionable, appropriately scoped |
| **Harmlessness** | No dangerous advice, toxic content, or privacy violations |
| **Groundedness** | Claims traceable to sources in provided context |
| **Refusal Correctness** | Refuses harmful requests; doesn't over-refuse benign ones |
| **Format Adherence** | Matches specified structure, syntax, tone, length |
| **Consistency** | No self-contradictions across turns or queries |

---

## License

This checklist is provided for educational and practical use. Adapt as needed for your application.
