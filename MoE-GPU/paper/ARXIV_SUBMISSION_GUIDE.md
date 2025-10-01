# arXiv Submission Guide - Expert-Sliced GPU Scheduling

Complete step-by-step guide for submitting your paper to arXiv.

## 📋 Pre-Submission Checklist

- [ ] Paper compiles without errors
- [ ] All authors have approved the submission
- [ ] Abstract is under 1920 characters
- [ ] References are properly formatted
- [ ] Figures are in PDF format (if any)
- [ ] No proprietary or confidential information
- [ ] Acknowledgments section is complete

## 🚀 Quick Start

### 1. Compile the Paper

```bash
cd paper

# Option A: Using Make (recommended)
make

# Option B: Using pdflatex directly
pdflatex research_paper.tex
pdflatex research_paper.tex  # Run twice

# Option C: Using latexmk
latexmk -pdf research_paper.tex
```

### 2. Verify the PDF

Open `research_paper.pdf` and check:
- All sections render correctly
- Tables are properly formatted
- Equations display correctly
- No missing references
- Page numbers are correct

### 3. Create arXiv Package

```bash
# Using Make
make arxiv

# This creates: arxiv_submission.tar.gz
```

### 4. Submit to arXiv

Go to: https://arxiv.org/submit

## 📝 Detailed Submission Steps

### Step 1: Create arXiv Account

1. Go to https://arxiv.org/
2. Click "register" (top right)
3. Complete registration with institutional email
4. Verify your email address
5. Complete your profile

### Step 2: Prepare Submission Files

#### Required Files:
- `research_paper.tex` - Main LaTeX source
- `research_paper.bbl` - Bibliography (if using BibTeX)

#### Optional Files:
- `figures/*.pdf` - Any figures you add
- `*.sty` - Custom style files (if any)

#### Create the Package:

```bash
# Manual method
mkdir arxiv_submission
cp research_paper.tex arxiv_submission/
# Add any figures if you have them
# cp figures/*.pdf arxiv_submission/

cd arxiv_submission
tar -czf ../arxiv_submission.tar.gz *
cd ..
```

### Step 3: Upload to arXiv

1. **Log in** to arXiv.org
2. Click **"Submit"** in the top menu
3. Click **"Start New Submission"**

### Step 4: Upload Files

1. Click **"Upload Files"**
2. Select `arxiv_submission.tar.gz`
3. Click **"Process"**
4. Wait for arXiv to process (usually < 1 minute)
5. Check for any errors or warnings

### Step 5: Select Categories

**Primary Category:** (Choose one)
- `cs.LG` - Machine Learning (RECOMMENDED)
- `cs.DC` - Distributed, Parallel, and Cluster Computing

**Cross-List Categories:** (Optional, but recommended)
- `cs.PF` - Performance
- `cs.AR` - Hardware Architecture
- `cs.AI` - Artificial Intelligence

**Why these categories?**
- `cs.LG`: Core ML research
- `cs.DC`: GPU scheduling and parallelism
- `cs.PF`: Performance optimization
- `cs.AR`: Hardware-aware design

### Step 6: Enter Metadata

#### Title:
```
Expert-Sliced GPU Scheduling: Dynamic Resource Allocation for Mixture of Experts Models
```

#### Authors:
```
Your Name (Your Institution)
Co-Author Name (Their Institution)
```

**Format:**
- First name, Last name
- One author per line
- Include institutional affiliation

#### Abstract:
```
Mixture of Experts (MoE) models achieve state-of-the-art performance across various domains but suffer from poor GPU utilization due to sparse expert activation patterns. We propose Expert-Sliced GPU Scheduling, a novel system that dynamically partitions GPU resources based on runtime profiling of expert usage. Our approach combines three key innovations: (1) Triton-optimized kernels for fused expert computation, (2) CUDA graph-based execution for reduced kernel launch overhead, and (3) dynamic GPU slice allocation aligned with expert routing sparsity. Experiments on NVIDIA A100/H100 GPUs demonstrate 1.8-2.4× throughput improvement and 35-45% energy efficiency gains compared to standard MoE implementations, while maintaining model accuracy.
```

**Note:** Abstract must be under 1920 characters (currently ~730)

#### Comments: (Optional)
```
17 pages, 6 tables, 2 algorithms. Code available at https://github.com/yourusername/moe-gpu-scheduling
```

### Step 7: Review and Submit

1. **Preview** the rendered PDF
2. Check all metadata is correct
3. Review the license (default: arXiv.org perpetual license)
4. Click **"Submit"**

### Step 8: Moderation

- arXiv moderators review submissions (usually 24-48 hours)
- You'll receive an email when approved
- Paper will be published the next business day

## 📅 arXiv Schedule

**Submission Deadlines** (US Eastern Time):
- **Sunday - Thursday**: 14:00 (2 PM) → Announced next day
- **Friday**: 14:00 (2 PM) → Announced Monday
- **Saturday**: No submissions

**Announcement Times:**
- Papers announced at 20:00 (8 PM) US Eastern Time

**Example Timeline:**
- Submit Monday 10:00 AM → Announced Tuesday 8:00 PM
- Submit Friday 1:00 PM → Announced Monday 8:00 PM

## 🔧 Troubleshooting

### Error: "TeX capacity exceeded"

**Solution:** Your paper is too large. Try:
```latex
% Add to preamble
\usepackage[draft]{graphicx}  % Don't include images
```

### Error: "File not found: XXX.sty"

**Solution:** Include the style file in your submission or use standard packages

### Error: "Undefined control sequence"

**Solution:** Check all custom commands are defined in the preamble

### Warning: "Overfull \hbox"

**Solution:** These are usually safe to ignore. They mean text extends slightly into margins.

### Error: "Cannot compile"

**Solution:** Test locally first:
```bash
pdflatex research_paper.tex
# Check the .log file for errors
```

## 📊 After Submission

### Your Paper URL

Once published, your paper will be at:
```
https://arxiv.org/abs/YYMM.NNNNN
```

Where:
- `YY` = Year (e.g., 25 for 2025)
- `MM` = Month (e.g., 01 for January)
- `NNNNN` = Paper number

**Example:** `https://arxiv.org/abs/2501.12345`

### Versions

You can submit updated versions:
1. Go to your paper page
2. Click "Replace"
3. Upload new version
4. Add version comment (e.g., "Fixed typos, updated results")

**Version URLs:**
- v1: `https://arxiv.org/abs/2501.12345v1`
- v2: `https://arxiv.org/abs/2501.12345v2`

### Citation

After publication, cite as:

```bibtex
@article{yourname2025expert,
  title={Expert-Sliced GPU Scheduling: Dynamic Resource Allocation for Mixture of Experts Models},
  author={Your Name and Co-Authors},
  journal={arXiv preprint arXiv:2501.12345},
  year={2025}
}
```

## 📢 Promoting Your Paper

### 1. Social Media

**Twitter/X Template:**
```
🚀 New paper on arXiv! "Expert-Sliced GPU Scheduling"

We achieve 2.3× speedup for MoE models through dynamic GPU resource allocation

✅ Triton kernels
✅ CUDA graphs  
✅ Energy efficient

Paper: https://arxiv.org/abs/2501.12345
Code: https://github.com/yourusername/moe-gpu-scheduling

#MachineLearning #GPU #MoE
```

### 2. Reddit

Post to:
- r/MachineLearning (use [R] tag for research)
- r/CUDA
- r/GPU

### 3. Mailing Lists

- ML News (https://mlnews.org/)
- Papers with Code (auto-indexed)

### 4. Conferences

Consider submitting to:
- **MLSys** (Machine Learning and Systems)
- **PPoPP** (Principles and Practice of Parallel Programming)
- **SC** (Supercomputing)
- **EuroSys** (European Conference on Computer Systems)
- **ASPLOS** (Architectural Support for Programming Languages and Operating Systems)

## 📋 Submission Checklist

Before clicking "Submit":

- [ ] Paper compiles without errors
- [ ] PDF looks correct (no missing text, figures, tables)
- [ ] All authors listed correctly
- [ ] Abstract is complete and under 1920 characters
- [ ] Categories selected (primary + cross-lists)
- [ ] Comments field filled (optional but recommended)
- [ ] License accepted (arXiv.org perpetual license)
- [ ] All co-authors have approved submission
- [ ] Code repository is public (if mentioned in paper)
- [ ] No confidential information in paper

## 🎯 Tips for Success

1. **Submit Early in the Week**: Avoid Friday/weekend submissions
2. **Check Before Deadline**: Submit before 2 PM ET to make next day's announcement
3. **Use Standard Packages**: Avoid custom LaTeX packages when possible
4. **Test Compilation**: Compile locally multiple times before submitting
5. **Proofread**: Have co-authors review before submission
6. **Include Code**: Link to GitHub repository increases citations
7. **Write Good Abstract**: First 250 chars appear in listings
8. **Choose Categories Carefully**: Affects who sees your paper

## 📞 Getting Help

### arXiv Support
- Email: help@arxiv.org
- Help pages: https://arxiv.org/help

### LaTeX Issues
- TeX StackExchange: https://tex.stackexchange.com/
- Overleaf Documentation: https://www.overleaf.com/learn

### Our Repository
- GitHub Issues: https://github.com/yourusername/moe-gpu-scheduling/issues

## 🎉 Success!

Once your paper is published:

1. ✅ Update your CV/website
2. ✅ Share on social media
3. ✅ Add to Papers with Code
4. ✅ Submit to conferences
5. ✅ Respond to feedback/questions

**Congratulations on your arXiv submission!** 🚀

---

*Last Updated: 2025-09-30*
*For questions: your.email@institution.edu*
