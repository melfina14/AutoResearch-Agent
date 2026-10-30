# AutoResearch Agent
Multi-Agent Scientific Literature Review and Hypothesis Generation Framework.

## Phase 1 goal
Run an end-to-end research workflow:
Research question -> planning -> Semantic Scholar retrieval -> paper analysis -> comparison -> gap detection -> hypothesis generation -> report.

## Setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```
Add your Gemini API key to `.env`, then:
```powershell
streamlit run frontend.py
```

Semantic Scholar's public Academic Graph API can be used without a key at low volume; an API key may be added to `.env` for higher limits.

## Development phases
1. Foundation + UI
2. Academic retrieval
3. PDF processing
4. RAG evidence layer
5. Specialized agents
6. LangGraph orchestration
7. Comparison + gap detection
8. Hypothesis generation/evaluation
9. Citation verification
10. Report generation
11. Evaluation + testing
12. Final documentation/demo
