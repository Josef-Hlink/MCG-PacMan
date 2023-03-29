# Pac-Man Competition

Assignment 2 for the course _Modern Game AI Algorithms_ at the Leiden Institute of Advanced Computer Science (LIACS).
Implementation by: _J. Hamelink, T. Blom, S. Sharma_

## Works with

- Python 3.11.2
- NumPy 1.24.2

## Setup

```bash
git clone <this-repo>
# recommended (if it works): create a virtual environment
python3 -m venv venv; source venv/bin/activate
# requirements
pip install -r requirements.txt
cd src
# run
python3 main.py
```


## Heuristics

For both:
- quickest route out of home column (to safe strip)


When ghost (defender):
- distances to attacking pacman(s)


When pacman (attacker):
- distances (Maze) to closest (Manhattan) five food pellets
- safest route home (to safe strip)
