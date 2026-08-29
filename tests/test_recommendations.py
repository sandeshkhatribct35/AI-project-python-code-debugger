from src.recommendations import recommendations_from_warnings, unique_recommendations


def test_duplicate_display_suggestions_are_removed_preserving_first_item():
    warnings = [
        {
            "type": "syntax_error",
            "severity": "error",
            "message": "Python syntax could not be parsed.",
            "suggestion": "Check the reported line.",
        },
        {
            "type": "unmatched_delimiter",
            "severity": "warning",
            "message": "Mismatched parentheses.",
            "suggestion": "Check the delimiter.",
        },
    ]

    recommendations = recommendations_from_warnings(warnings)
    displayed = unique_recommendations(recommendations)

    assert len(recommendations) == 2
    assert len(displayed) == 1
    assert displayed[0]["type"] == "syntax_error"
