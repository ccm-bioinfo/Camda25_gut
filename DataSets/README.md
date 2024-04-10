Here we provide a dataset consisting of **613 patients from 3 different cohorts** (HMP2 and two AGP cohorts). There are 3 categories of individuals:

* Healthy,
* Obese, and
* with IBD (one of two subtypes, CD = Crohn’s disease, UC = ulcerative colitis).

We provide 3 files:

* **taxonomy.txt**: species-level contribution to the taxonomic profile, calculated using MetaPhlAn
* **pathways.txt**: contributions of different species to pathways, calculated using HumanN. “UNINTEGRATED” stands for unassigned pathways.
* **metadata.txt**: contains sample names (raw fastq files can be identified in SRA by sample name), project and diagnosis assignment for each sample, along with scores predicted by the existing taxonomic health indices. Note that higher scores indicate better health for Shannon entropy and GMHI, while worse disease for hiPCA.

The dataset has not been filtered in any way, in order for you to decide on the best method to filter it based on your solution.