# WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient-Facing Dialogue

A benchmark, alignment pipeline, and LLM-as-a-Judge for evaluating the clinical impact of ASR errors.

---

## 🔍 Overview

This repository hosts the code, models, and datasets accompanying the paper.
The work investigates how Automatic Speech Recognition (ASR) errors distort clinical meaning in patient-facing dialogue — and shows that traditional metrics like Word Error Rate (WER) fail to capture real clinical risk.

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

Preprint available on arXiv:
https://arxiv.org/abs/2511.16544

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
