from pathlib import Path
import json
import os

from openai import OpenAI
import quorum_translate as q

q.load_dotenv(Path('.env'))
api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
if not api_key:
    raise SystemExit('Missing OPENROUTER_API_KEY (or OPENAI_API_KEY).')

client = OpenAI(api_key=api_key, base_url=q.OPENROUTER_BASE_URL)

preference = (
    'This should be readable by a 7 year old. '
    'Use smooth plain clauses, and express plausibility in natural terms like whether something sounds believable or could really happen.'
)

# Keep reference alignment with the selected Greek paragraph.
q.DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS = [q.DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS[2]]
q.DEFAULT_PERRIN_PARAGRAPHS = [q.DEFAULT_PERRIN_PARAGRAPHS[2]]

result = q.run_pipeline(
    client=client,
    model='x-ai/grok-4.1-fast',
    greek_paragraphs=[q.DEFAULT_GREEK_PARAGRAPHS[2]],
    iterations=int(os.getenv('P3_TUNE_ITERS', '4')),
    verbose=True,
    color_mode='always',
    user_preference=preference,
    pipeline='sequential',
)

prefix = Path(os.getenv('P3_TUNE_PREFIX', 'p3_tune_run9'))
json_path = prefix.with_suffix('.json')
md_path = prefix.with_suffix('.md')
json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
md_path.write_text(q.render_markdown_report(result), encoding='utf-8')
print(result['final_translation'])
print()
print(f'Wrote {json_path}')
print(f'Wrote {md_path}')
