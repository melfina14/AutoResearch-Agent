
from app.agents.retrieval import retrieval_agent

question = "Blockchain technology in supply chain management"

plan = {
    "research_topic": question,
    "search_queries": [question],
}

papers = retrieval_agent(question, plan, limit=5)

print("\nPAPERS RETURNED:", len(papers))

for index, paper in enumerate(papers, start=1):
    print(f"\n{index}. {paper.get('title')}")
    print("Source:", paper.get("source"))
    print("Abstract:", (paper.get("abstract") or "Not available")[:350])
