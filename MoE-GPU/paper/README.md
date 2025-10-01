# Research Paper - Expert-Sliced GPU Scheduling

This directory contains the research paper in both Markdown and LaTeX formats.

## Files

- **`research_paper.md`** - Markdown version (for GitHub/web viewing)
- **`research_paper.tex`** - LaTeX source (for arXiv submission)
- **`README.md`** - This file

## Compiling the LaTeX Paper

### Prerequisites

You need a LaTeX distribution installed:
- **Windows**: MiKTeX or TeX Live
- **macOS**: MacTeX
- **Linux**: TeX Live

### Compilation Commands

#### Option 1: Using pdflatex (Recommended)

```bash
cd paper
pdflatex research_paper.tex
pdflatex research_paper.tex  # Run twice for references
```

#### Option 2: Using latexmk (Automated)

```bash
cd paper
latexmk -pdf research_paper.tex
```

#### Option 3: Using Overleaf

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project
3. Upload `research_paper.tex`
4. Compile online

### Output

The compilation will generate:
- `research_paper.pdf` - The final paper
- `research_paper.aux`, `.log`, `.out` - Auxiliary files (can be deleted)

## Preparing for arXiv Submission

### Step 1: Compile Locally

```bash
pdflatex research_paper.tex
pdflatex research_paper.tex
```

### Step 2: Check the PDF

Open `research_paper.pdf` and verify:
- All sections are present
- Tables and equations render correctly
- References are properly formatted
- No compilation warnings

### Step 3: Prepare Submission Package

For arXiv, you need to submit the source files:

```bash
# Create a submission directory
mkdir arxiv_submission
cp research_paper.tex arxiv_submission/
cp research_paper.bbl arxiv_submission/  # If using BibTeX

# If you have figures (add them later):
# cp figures/*.pdf arxiv_submission/
```

### Step 4: Create arXiv-Compatible Archive

```bash
cd arxiv_submission
tar -czf ../arxiv_submission.tar.gz *
```

### Step 5: Submit to arXiv

1. Go to [arXiv.org](https://arxiv.org/)
2. Create an account or log in
3. Click "Submit" → "Start New Submission"
4. Upload `arxiv_submission.tar.gz`
5. Select category: **cs.LG** (Machine Learning) or **cs.DC** (Distributed Computing)
6. Add cross-lists: **cs.PF** (Performance), **cs.AR** (Hardware Architecture)
7. Fill in metadata (title, authors, abstract)
8. Submit for moderation

## arXiv Categories

**Primary Category:**
- `cs.LG` - Machine Learning

**Cross-List Categories:**
- `cs.DC` - Distributed, Parallel, and Cluster Computing
- `cs.PF` - Performance
- `cs.AR` - Hardware Architecture

## Paper Metadata

**Title:** Expert-Sliced GPU Scheduling: Dynamic Resource Allocation for Mixture of Experts Models

**Abstract:** (First 250 characters)
Mixture of Experts (MoE) models achieve state-of-the-art performance across various domains but suffer from poor GPU utilization due to sparse expert activation patterns. We propose Expert-Sliced GPU Scheduling, a novel system that dynamically partitions...

**Keywords:** Mixture of Experts, GPU Scheduling, Resource Allocation, CUDA Graphs, Triton Kernels, Energy Efficiency

## Customization

### Adding Authors

Edit the `\author{}` section in `research_paper.tex`:

```latex
\author{
    First Author\thanks{Equal contribution} \\
    Institution \\
    \texttt{email@institution.edu} \\
    \And
    Second Author$^*$ \\
    Institution \\
    \texttt{email@institution.edu}
}
```

### Adding Figures

1. Create a `figures/` directory
2. Add your plots (PDF format recommended)
3. Include in LaTeX:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\columnwidth]{figures/throughput_comparison.pdf}
    \caption{Throughput comparison between baseline and optimized MoE.}
    \label{fig:throughput}
\end{figure}
```

### Adding Tables

Tables are already included. To add more:

```latex
\begin{table}[h]
\centering
\caption{Your Table Caption}
\label{tab:yourlabel}
\begin{tabular}{lcc}
\toprule
Column 1 & Column 2 & Column 3 \\
\midrule
Data 1   & Data 2   & Data 3   \\
\bottomrule
\end{tabular}
\end{table}
```

## Converting Plots to PDF

If you have PNG plots from the visualization script:

```bash
# Install ImageMagick (if needed)
# Windows: choco install imagemagick
# macOS: brew install imagemagick
# Linux: sudo apt-get install imagemagick

# Convert PNG to PDF
convert plots/throughput_comparison.png figures/throughput_comparison.pdf
convert plots/gpu_utilization.png figures/gpu_utilization.pdf
# ... etc
```

Or use Python:

```python
from PIL import Image
import os

for png_file in os.listdir('plots'):
    if png_file.endswith('.png'):
        img = Image.open(f'plots/{png_file}')
        pdf_file = png_file.replace('.png', '.pdf')
        img.save(f'figures/{pdf_file}', 'PDF', resolution=300.0)
```

## Common LaTeX Issues

### Issue: "File not found"
**Solution:** Make sure all referenced files are in the same directory or use relative paths

### Issue: "Undefined control sequence"
**Solution:** Check that all required packages are installed in the preamble

### Issue: "Citation undefined"
**Solution:** Run pdflatex twice to resolve references

### Issue: "Overfull hbox"
**Solution:** These are warnings about text extending into margins. Usually safe to ignore for arXiv

## Checking for arXiv Compliance

arXiv has specific requirements:

1. **File size**: < 10 MB (usually not an issue for LaTeX)
2. **Compilation**: Must compile with pdflatex
3. **Fonts**: Use standard fonts (already done)
4. **Figures**: PDF or EPS format preferred
5. **Bibliography**: Can use BibTeX or inline bibliography (we use inline)

## Version Control

When updating the paper:

```bash
# Save a version
cp research_paper.tex research_paper_v1.tex

# Make changes to research_paper.tex

# Compile and check
pdflatex research_paper.tex

# If good, commit
git add research_paper.tex
git commit -m "Updated results section"
```

## Citation

Once published on arXiv, cite as:

```bibtex
@article{yourname2025expert,
  title={Expert-Sliced GPU Scheduling: Dynamic Resource Allocation for Mixture of Experts Models},
  author={Your Name and Co-Authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Contact

For questions about the paper:
- Email: your.email@institution.edu
- GitHub Issues: https://github.com/yourusername/moe-gpu-scheduling/issues

## License

This paper is released under CC BY 4.0 (Creative Commons Attribution 4.0 International)
