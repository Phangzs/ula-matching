# Equitable TA Assignment with Incomplete Preferences: Mixed-Experience Constraints via ILP and a Gale-Shapley Baseline

## Abstract
Assigning teaching assistants (TAs) to courses is often a manual, tedious process, and overlooks equity goals such as mixing novice and veteran staff. We study the TA-matching problem in a university computer-science department where preference data are highly incomplete: (i) students list only their top-two courses and tiered enthusiasm for the rest, (ii) professors rank a limited subset proportional to the course quota.

As a baseline we run the student-proposing Gale-Shapley algorithm on the induced preference lists. Because many stable matchings exist, we sample random student orders and apply a greedy tie-breaker to return a canonical stable outcome.

To incorporate equity, we formulate an integer linear program (ILP) that maximizes total preference score subject to a mixed-experience constraint: any course requiring ≥ 2 TAs must receive at least one first-time and one experienced TA. The ILP template also supports arbitrary binary fairness constraints, enabling extensions to workload or diversity goals.

We will release: (i) open-source code for the Gale-Shapley sampling and the ILP model, (ii) an empirical evaluation comparing preference satisfaction and runtime on real semesters, and (iii) a sample of an anonymized dataset of student and professor preferences, showing typical categories and values.

Our work shows how domain-driven constraints can convert a classical stable-matching formulation into a deployable, fairness-aware assignment tool for higher-education settings.
