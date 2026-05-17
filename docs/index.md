# concept-graph-xai

**SHAP, told the way your model committee actually argues.**

---

## The gap

A SHAP plot tells you which **features** moved a prediction. A model
risk committee, a fairness review, or a regulator asks something
different: *which concept did the model lean on, and was that concept
the right thing to lean on at all?*

`monthly_income`, `debt_ratio`, and `revolving_utilization` are not
three independent levers — they are three windows onto **Income +
Utilization**. `n_30_59_dpd`, `n_60_89_dpd`, `n_90_plus_dpd` are not
three independent features — they are one **Delinquency** signal at
three time horizons. The hierarchy that matters is the one your
business, your ontology, and your regulator argue in. SHAP does not
know about that hierarchy. Auto-clustering tools (`shap.utils.hclust`,
`seaborn.clustermap`) infer a different hierarchy from the data — one
that maximises correlation, not one that mirrors *PII vs. financial
vs. behavioural vs. bureau-supplied*.

That is the gap. You already have the hierarchy on a whiteboard, in a
regulatory submission, in a feature-store taxonomy. `concept-graph-xai`
takes that hierarchy as input and reshapes every per-feature
interpretability signal — importance, utilization, interactions,
ablations, drift, per-group fairness — through it.

## What you get

![Concept-coherence vs. concept-importance](img/coherence_importance.png){ width="640" }

The single chart that no per-feature SHAP plot can produce: every
concept in your tree placed on *how coherent its features are* against
*how much the model leans on it*. The top-left quadrant is the danger
zone — concepts the model relies on heavily but whose features look
like three different things. Those are the concepts to split before
the next review meeting.

The same engine produces concept-level versions of everything else:
importance and utilization sunbursts, SHAP interaction matrices,
Sankey flow from features through the tree to the decision, per-row
explanations rolled up to concepts, segment / cohort / protected-group
slices, ablation under whole-branch outages, and across-period drift.

## Start here

[Take the tour :material-arrow-right:](tour.md){ .md-button .md-button--primary }
[Install + hello world :material-arrow-right:](getting-started.md){ .md-button }

The Tour walks one realistic credit-risk scenario end-to-end, from
training a model to defending its concept structure. If you prefer to
start from the API surface, the [User Guide](user-guide/concept-graphs.md)
indexes the seven questions the library is designed to answer, one
page each.

---

*Status: alpha (v0.6.1). The metric layer (`metrics/*`) and plot layer
(`plotting/*`) are stable and independently testable; v1.0 will add
DAG support with optional per-edge weights. See the
[roadmap](roadmap.md) for the milestone view.*
