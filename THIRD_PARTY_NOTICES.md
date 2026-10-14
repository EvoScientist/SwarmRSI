# Third-party notices

SwarmRSI is released under the Apache-2.0 license. The package name `evolvcoder` is preserved.

The baseline fetcher retrieves independently licensed upstream projects; their sources and datasets are not redistributed in this repository. Preserve each upstream checkout's LICENSE: AIDE (WecoAI/aideml), Arbor (RUC-NLPIR/Arbor), MLEvolve (InternScience/MLEvolve), and TusoAI (Alistair-Turcan/TusoAI). Exact revisions are in `baselines/upstreams.json`.

NatureBench and MLE-Bench task data and official graders must be obtained from their upstream projects under their respective terms. MLE-Bench grader custody in this project was pinned to openai/mle-bench commit `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`.

`src/evolvcoder/openproblems_tasks.py` restates OpenProblems denoising's binomial molecule split, originating in czbiohub/molecular-cross-validation (MIT, Copyright 2019 Chan Zuckerberg Biohub), and its normalized log-count MSE. The external six-task adapter uses output-only reconstructed metrics; the README states their scope. The historical dimensionality-reduction component formulas follow OpenProblems commit `ab268cc0d6781cee7140a73b3943761104719ad8`; the normalized aggregate is explicitly a reconstruction.

Scientific libraries are installed separately and retain their own licenses. This release contains no pretrained weights, data, article PDFs, or third-party agent-generated solutions.
