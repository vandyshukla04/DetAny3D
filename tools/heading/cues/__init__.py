"""Head cues: image evidence that answers 'which horizontal face is the front?'.

Each cue implements `HeadCue` (see base.py) and may ABSTAIN when it has no evidence.
"""
from tools.heading.cues.base import CueContext, HeadCue, projected_face_centers

__all__ = ["CueContext", "HeadCue", "projected_face_centers"]
