"""
Three dashboard features that share one underlying data structure, built
together rather than as separate modules per the roadmap decision:
  1. risk_level()        -> green/yellow/red from the similarity score
  2. transition_note()   -> plausibility note for a (class_t1, class_t2) pair
  3. generate_explanation() -> the human-readable narrative combining both

All of this is explicitly RULE-BASED / TEMPLATED - no learned model, no
LLM. Label it that way in the report; don't call it "AI-generated."
"""

# A small, illustrative plausibility table. Extend as needed - this is
# intentionally not exhaustive, since exhaustive coverage of all 10x10
# class-pair combinations adds little value over the illustrative cases
# most likely to come up in a EuroSAT demo.
TRANSITION_PLAUSIBILITY = {
    ("Forest", "Residential"): "common - consistent with urban expansion into vegetated land",
    ("Forest", "Industrial"): "common - consistent with land clearing for development",
    ("AnnualCrop", "Residential"): "common - consistent with agricultural land converted to housing",
    ("Pasture", "Residential"): "common - consistent with suburban expansion",
    ("Residential", "Forest"): "uncommon - would imply land reverting to vegetation, verify manually",
    ("SeaLake", "Highway"): "implausible - flag for manual review, likely a misclassification",
    ("SeaLake", "Residential"): "implausible - flag for manual review, likely a misclassification",
    ("River", "Industrial"): "possible - consistent with riverside industrial development",
}


def risk_level(similarity, stable_min=0.85, moderate_min=0.65):
    """Returns one of 'stable', 'moderate', 'significant' plus a color."""
    if similarity >= stable_min:
        return {"level": "stable", "color": "green", "emoji": "🟢"}
    elif similarity >= moderate_min:
        return {"level": "moderate", "color": "yellow", "emoji": "🟡"}
    else:
        return {"level": "significant", "color": "red", "emoji": "🔴"}


def transition_note(class_t1, class_t2):
    if class_t1 == class_t2:
        return "No class change detected between T1 and T2."
    note = TRANSITION_PLAUSIBILITY.get((class_t1, class_t2))
    if note is None:
        note = "no specific pattern on record for this class pair - treat as a general land-use change"
    return note


def generate_explanation(class_t1, class_t2, similarity, stable_min=0.85, moderate_min=0.65):
    """Builds the dashboard-facing narrative. Templated, not generative -
    see module docstring."""
    risk = risk_level(similarity, stable_min, moderate_min)
    note = transition_note(class_t1, class_t2)

    if class_t1 == class_t2:
        headline = f"Detected {risk['level'].capitalize()} Change"
        interpretation = f"Land-use class remained '{class_t1}'. {note}"
    else:
        headline = f"Detected {risk['level'].capitalize()} Change"
        interpretation = f"{note.capitalize()}."

    return {
        "headline": headline,
        "previous_class": class_t1,
        "current_class": class_t2,
        "similarity": round(similarity, 3),
        "risk_level": risk["level"],
        "risk_color": risk["color"],
        "risk_emoji": risk["emoji"],
        "interpretation": interpretation,
    }
