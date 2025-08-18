import arxiv
import os
import json

output_dir = "data/pdfs"
os.makedirs(output_dir, exist_ok=True)

search = arxiv.Search(
    query="cat:cs.CL",
    max_results=50,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

meta_file = "data/meta.jsonl"
with open(meta_file, "w", encoding="utf-8") as f_meta:
    for result in search.results():
        pdf_path = os.path.join(output_dir, f"{result.entry_id.split('/')[-1]}.pdf")
        result.download_pdf(dirpath=output_dir, filename=os.path.basename(pdf_path))
        meta = {
            "id": result.entry_id,
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "pdf_path": pdf_path
        }
        f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"Downloaded: {result.title}")
