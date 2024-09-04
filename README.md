# LogoFinder
Logo finder using GroundingDino and OpenCV
Grounding DINO used to find generic 'logo' etc class,
SIFT keypoint matching in OpenCV used to test logos found against a reference image


# Setup
```
git clone https://github.com/oscr104/LogoFinder.git
cd LogoFinder
conda create --name LogoFinder python=3.12
conda activate LogoFinder
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -q -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
pip install -r requirements.txt
```