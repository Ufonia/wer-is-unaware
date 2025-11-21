# WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient-Facing Dialogue

A benchmark, alignment pipeline, and LLM-as-a-Judge for evaluating the clinical impact of ASR errors.

---

## 🔍 Overview

This repository will host the code, models, and datasets accompanying the paper.
The work investigates how Automatic Speech Recognition (ASR) errors distort clinical meaning in patient-facing dialogue — and shows that traditional metrics like Word Error Rate (WER) fail to capture real clinical risk.

We introduce:
- A clinician-annotated benchmark of ASR errors labelled for clinical impact
- A semantic LLM-based aligner for robust ground-truth ↔ ASR utterance alignment
- An LLM-as-a-Judge, optimized with GEPA, that achieves human-comparable performance
- Evaluations of 20+ ASR metrics, showing their poor correlation with clinical safety
- All resources will be made available here soon.

---

 ## 📦 Coming Soon

This repository will be populated with:

### Dataset release:
Clinician-labelled clinical-impact benchmark
(Primock57-derived subset + metadata)

### Code:

- LLM-based alignment pipeline
- GEPA-optimized clinical-risk evaluator
- Metric evaluation scripts
- Reproducible evaluation pipeline


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
