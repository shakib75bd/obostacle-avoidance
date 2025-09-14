# Research Paper: Real-Time Monocular Obstacle Avoidance

This folder contains all files related to the academic research paper on uncertainty-guided adaptive region fusion for obstacle avoidance.

## 📄 Paper Files

### Main Documents
- `research_paper.tex` - Main LaTeX source file (25 pages, single-column format)
- `research_paper.pdf` - Compiled PDF version of the paper
- `README_PAPER.md` - Detailed documentation about the paper structure and content

### Supporting Documents
- `obstacle-avoidance.pdf` - Original project document
- `obstacle-avoidance.txt` - Original project text description

## 🖼️ Figures and Images

All figures used in the research paper:
- `system_architecture.png` - Overall system architecture diagram
- `timing_breakdown.png` - Processing time distribution analysis
- `uncertainty_analysis.png` - Uncertainty quantification performance
- `safety_analysis.png` - Safety performance distribution
- `yolov8_comparison_table.png` - YOLOv8 performance comparison

## 🔧 LaTeX Compilation Files

Generated during compilation (can be safely deleted):
- `research_paper.aux` - Auxiliary file for cross-references
- `research_paper.fdb_latexmk` - LaTeXmk database file
- `research_paper.fls` - File list for LaTeX
- `research_paper.log` - Compilation log
- `research_paper.out` - Hyperref outline file
- `research_paper.synctex.gz` - SyncTeX file for editor integration

## 📝 How to Compile

To compile the LaTeX document:

```bash
cd paper/
pdflatex research_paper.tex
```

For complete compilation with references:
```bash
pdflatex research_paper.tex
pdflatex research_paper.tex
```

## 📊 Paper Statistics

- **Format**: Single-column academic paper
- **Length**: 25 pages
- **File Size**: ~1 MB
- **Figures**: 4 main figures + 1 comparison table
- **Tables**: 6 comprehensive performance tables
- **References**: 15 academic citations

## 🎯 Key Contributions

1. **Uncertainty-Guided Adaptive Fusion**: Novel approach achieving 7.4% improvement in navigation accuracy
2. **Real-Time Performance**: 24.5 FPS on consumer hardware
3. **Comprehensive Safety Analysis**: 4.8% false safe rate
4. **Modular Architecture**: Flexible, extensible framework
5. **Open-Source Implementation**: Complete reproducible research

## 📋 Paper Structure

1. **Abstract & Introduction** - Problem motivation and contributions
2. **Related Work** - Literature review and positioning
3. **Methodology** - Detailed system architecture and algorithms
4. **Experimental Setup** - Implementation details and evaluation framework
5. **Results & Analysis** - Comprehensive performance evaluation
6. **Discussion** - Analysis of results and implications
7. **Conclusion** - Summary and future directions

## 🔗 Dependencies

Required for LaTeX compilation:
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Packages: amsmath, graphicx, booktabs, algorithm, hyperref, etc.

All required packages are standard and included in most LaTeX distributions.

## 📄 License

This research paper and associated materials are part of the obstacle avoidance system project. See the main project README for licensing information.
