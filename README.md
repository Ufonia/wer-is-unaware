# WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient-Facing Dialogue

As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's κ of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue.

---

## 🔍 Overview

This repository hosts the code, models, and datasets accompanying the paper.

We introduce (available here):
- Clinician-annotated clinical-impact dataset: `llm_judge/dataset/primock_data_final_outcomes.csv`
- Semantic LLM-based aligner: `alignment/aligner/` (see `alignment/README.md` for usage)
- LLM-as-a-Judge optimized with GEPA/MIPRO: `llm_judge/` (artifacts in `llm_judge/results/`)
- Evaluations of ASR metrics (code under `alignment/scripts/` and `alignment/results/`)

---

 ## 📦 Coming Soon

- Additional dataset metadata and documentation
- Evaluations of 20+ ASR metrics, showing their poor correlation with clinical safety

---

## 📄 Paper

Preprint available on arXiv: https://arxiv.org/abs/2511.16544

## Citation
```
@misc{ellis2025werunawareassessingasr,
      title={WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue}, 
      author={Zachary Ellis and Jared Joselowitz and Yash Deo and Yajie He and Anna Kalygina and Aisling Higham and Mana Rahimzadeh and Yan Jia and Ibrahim Habli and Ernest Lim},
      year={2025},
      eprint={2511.16544},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.16544}, 
}
```
