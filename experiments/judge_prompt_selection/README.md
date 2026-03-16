## Generate Responses

* Takes n prompts from the Ultrachat dataset
* Takes the attributes from `configs/attributes.yaml`
* Generates responses for each attribute and side (follow/avoid)

Usage:
```bash
python experiments/judge_prompt_selection/scripts/generate_responses.py
```

## Test Judge

A small experiment to evaluate the judge. Only uses the "follow" side attributes.
For each user, it randomly selects one attributes as "true preference". Then it judges each response against the "true preference" attribute. It then computes the fraction of users where the highest scored attribute is the "true preference" attribute.

Usage:
```bash
python experiments/judge_prompt_selection/scripts/test_judge.py
```

## Results
Judge: llm_personalization/judge/parsed_rating_judge.py
Judge Prompts: llm_personalization/judge/prompt_templates.py

Accuracy (GT gets highest score): 0.880 (88/100)
Accuracy strict (GT is sole max): 0.570 (57/100)
Average margin over mean: 3.282

(TODO: is the generator (prompt) good enough?)