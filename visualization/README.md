# CWMR-RL Manim Animation Instructions

This document explains how to run the Manim animation script that visualizes our CWMR-RL portfolio optimization project.

## About the Animation

The `cwmr_manim_explanation.py` script creates a comprehensive animated video that:

1. Explains the portfolio optimization problem
2. Demonstrates how mean reversion works in financial markets
3. Details the Confidence-Weighted Mean Reversion (CWMR) algorithm
4. Shows how we've enhanced CWMR with Reinforcement Learning
5. Compares the performance of different strategies:
   - GRPO-CWMR (0.70 Sharpe, 47% drawdown)
   - RL-PPO (0.60 Sharpe, 42% drawdown)
   - Multi-Agent Ensemble (0.67 Sharpe, 36% drawdown)
   - Equal Weight (6% return)
   - Original CWMR (12-13% return)
6. Explores future research directions and implications

The animation is designed to be educational and informative, making complex financial concepts accessible.

## Requirements

To run this animation, you'll need:

1. **Python 3.7+**
2. **Manim** - The Mathematical Animation Engine

### Installing Manim

Manim has several dependencies including LaTeX. Follow these steps:

#### MacOS Installation

```bash
# Install homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install py3cairo ffmpeg

# Install LaTeX (might take a while)
brew install --cask mactex

# Install Manim via pip
pip install manim
```

#### Linux Installation

```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg texlive texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-science tipa libcairo2-dev

# Install Manim via pip
pip install manim
```

#### Windows Installation

For Windows, the recommended approach is to use the community version:

```bash
pip install manim
```

You'll also need to install FFmpeg and LaTeX separately:
- FFmpeg: Download from https://ffmpeg.org/download.html
- LaTeX: Install MiKTeX from https://miktex.org/download

## Running the Animation

Once Manim is installed, you can generate the animation with:

```bash
# Navigate to the project directory
cd /Users/ryanmathieu/RL-CwMR

# Generate high-quality animation
manim -pqh cwmr_manim_explanation.py CWMRRLExplanation

# Generate medium quality (faster)
manim -pqm cwmr_manim_explanation.py CWMRRLExplanation

# Generate low quality (for quick testing)
manim -pql cwmr_manim_explanation.py CWMRRLExplanation
```

The `-p` flag plays the animation after rendering, and the `q` flag followed by `l`, `m`, or `h` sets the quality level.

## Animation Output

The rendered animation will be saved in the `media/videos/cwmr_manim_explanation/1080p60/` directory with the name `CWMRRLExplanation.mp4`.

## Customizing the Animation

If you want to customize aspects of the animation:

- **Colors**: Modify the color constants in the script (GREEN, BLUE, RED, etc.)
- **Timing**: Adjust the `run_time` parameters in the `play` functions
- **Text**: Edit the text content within the `Text` objects
- **Data**: Modify the performance metrics based on updated results

## Troubleshooting

If you encounter issues:

1. **Missing dependencies**: Ensure all required packages are installed
2. **LaTeX errors**: Check that your LaTeX installation is complete
3. **Memory issues**: For high-quality renders, ensure your system has sufficient RAM

## Credits

This animation was created using Manim, an animation engine developed by Grant Sanderson (3Blue1Brown). The content showcases our CWMR-RL portfolio optimization project, highlighting the advantages of reinforcement learning and multi-agent approaches in financial markets. 