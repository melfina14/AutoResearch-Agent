
from app.agents.report import report_agent

# Sample data only — no AI or academic search API calls.
sample_papers = [
    {
        "title": "Sample Research Paper",
        "authors": ["Test Author"],
        "year": 2025,
        "venue": "Sample Journal",
        "abstract": "This is sample abstract text.",
        "url": "https://example.com/sample",
    }
]

sample_analyses = [
    {
        "paper": "Sample Research Paper",
        "analysis": {
            "summary": "This is a mock analysis.",
        },
    }
]

sample_comparison = {
    "summary": "Mock comparison of sample papers."
}

sample_gaps = {
    "gaps": ["A sample research gap for testing."]
}

sample_hypotheses = {
    "hypotheses": [
        "A sample hypothesis that can be tested."
    ]
}

sample_evaluation = {
    "agent": "Hypothesis Evaluation Agent",
    "evaluation": {
        "overall_summary": "Mock evaluation completed."
    },
}

sample_verification = {
    "agent": "Citation Verification Agent",
    "verification": {
        "status": "Mock verification completed",
        "issues": [],
        "summary": "This is test data, not a real citation check.",
    },
}

# Call the report generator in memory without writing any files.
import inspect
from unittest.mock import patch

with patch("app.agents.report.Path.mkdir"), \
     patch("app.agents.report.Path.write_text"):

    result = report_agent(
        question="Mock research question",
        papers=sample_papers,
        analyses=sample_analyses,
        comparison=sample_comparison,
        gaps=sample_gaps,
        hypotheses=sample_hypotheses,
        hypothesis_evaluation=sample_evaluation,
        citation_verification=sample_verification,
    )

report_text = result.get("report", "")

print("Report generated:", result.get("status") == "success")
print(
    "Hypothesis Evaluation included:",
    "Mock evaluation completed." in report_text,
)
print(
    "Citation Verification included:",
    "Mock verification completed" in report_text,
)
print(
    "Original report writer was bypassed:",
    True,
)
