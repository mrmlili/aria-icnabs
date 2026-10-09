# ARIA-ICNABS

A Diagnostic Framework for Detecting False Fractionalization in Physics-Informed Neural Networks.

**Author:** Maryam Jalili (mrmjlili@iau.ac.ir)
**Affiliation:** Department of Mathematics, Neyshabur Branch, Islamic Azad University, Neyshabur, Iran

## Overview

This repository contains code, data, and manuscript for a study on fractional
derivative discretizations in Physics-Informed Neural Networks (PINNs).

We introduce a three-test diagnostic protocol (kernel-presence, dt-refinement,
effective-time) to detect whether a PINN residual genuinely preserves the
non-local memory of the fractional operator.

## Key results (alpha = 0.8, 3 seeds)

| Residual | MSE (mean +/- std) | Time (s) |
|----------|---------------------|----------|
| Naive FD | 3.04e-6 +/- 1.42e-6 | 216 |
| Caputo-L1 | 6.19e-6 +/- 0.72e-6 | 2327 |
| ABC-L1 | 4.19e-5 +/- 9.17e-6 | 2968 |

At alpha = 0.5, the ranking changes: Caputo-L1 outperforms Naive FD.

## Repository structure

- paper/ : LaTeX source and PDF
- code/ : Python scripts
- data/ : Reference solutions and PINN results (.npz files)
- figures/ : figure PDFs

## IMPORTANT: Pre-computed data included

All training results and reference solutions are already included in data/.
You do NOT need to re-run training scripts unless reproducing from scratch.

To regenerate figures, run the fig*.py scripts in code/.

## Requirements

- Python 3.10+
- PyTorch (CPU), NumPy, SciPy, Matplotlib

Install: pip install torch numpy scipy matplotlib

## Compiling the paper

    cd paper
    pdflatex ARIA_ICNABS_v14.tex
    pdflatex ARIA_ICNABS_v14.tex

## Citation

If you use this code or data, please cite:

    @article{Jalili2026ARIA,
      author = {Jalili, Maryam},
      title  = {A Diagnostic Framework for Detecting False Fractionalization in Physics-Informed Neural Networks},
      year   = {2026},
      note   = {Manuscript submitted for publication}
    }

## License

MIT License - see LICENSE file.

## Contact

Maryam Jalili - mrmjlili@iau.ac.ir
