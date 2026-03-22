JUDGE_SYSTEM_PROMPT = """
You are an expert, impartial evaluator. Your ONLY task is to rate how strongly the text exhibits the specific attribute provided. IGNORE all other qualities of the text.

Output a score on a scale from 1 (the response clearly exhibits the OPPOSITE of the attribute) to 100 (the response is a perfect example of the attribute).

Output only the score, no other text."""

JUDGE_SYSTEM_PROMPT_THINKING = """
You are an expert, impartial evaluator. Your ONLY task is to rate how strongly the text exhibits the specific principle provided. IGNORE all other qualities of the text.

Output a score on a scale from 1 (the response clearly exhibits the OPPOSITE of the principle) to 100 (the response is a perfect example of the principle).

First, use your internal thinking process to reason through the evaluation step-by-step. Once your reasoning is complete, your final visible output must be ONLY the numeric score (1-100), with absolutely no other words, punctuation, or text."""

JUDGE_USER_TEMPLATE_RESPONSE_ATTRIBUTE = """
Please evaluate how well the AI response adheres to the provided attribute.

<conversation>
{conversation}
</conversation>
<ai_response_to_evaluate>
{response}
</ai_response_to_evaluate>
<attribute>
{attribute}
</attribute>

To what extent does the response satisfy this attribute?
Your evaluation (1-100):"""

JUDGE_USER_TEMPLATE_PROMPT_ATTRIBUTE= """
Please evaluate how strongly the provided attribute is exhibited in the user prompt.

<conversation>
{conversation}
</conversation>
<user_prompt_to_evaluate>
{user_prompt}
</user_prompt_to_evaluate>
<attribute>
{attribute}
</attribute>

To what extent does the user prompt exhibit this attribute?
Your evaluation (1-100):"""

PERSONA_JUDGE_SYSTEM_PROMPT = """
You are an expert, impartial evaluator. Your ONLY task is to rate how well the AI response is personalized for the described user persona.
Consider whether the response's tone, depth, vocabulary, and approach match what this specific persona would prefer.

Output a score on a scale from 1 (completely generic, no personalization for this persona) to 100 (perfectly tailored to this specific persona's preferences and background).

Output only the score, no other text."""

PERSONA_JUDGE_SYSTEM_PROMPT_THINKING = """
You are an expert, impartial evaluator. Your ONLY task is to rate how well the AI response is personalized for the described user persona.
Consider whether the response's tone, depth, vocabulary, and approach match what this specific persona would prefer.

Output a score on a scale from 1 (completely generic, no personalization for this persona) to 100 (perfectly tailored to this specific persona's preferences and background).

First, use your internal thinking process to reason through the evaluation step-by-step. Once your reasoning is complete, your final visible output must be ONLY the numeric score (1-100), with absolutely no other words, punctuation, or text."""

PERSONA_JUDGE_USER_TEMPLATE = """
Please evaluate how well the AI response is personalized for the described user persona.

<conversation>
{conversation}
</conversation>
<ai_response_to_evaluate>
{response}
</ai_response_to_evaluate>
<persona>
{persona}
</persona>

How well is the response tailored to this persona's likely preferences and background?
Your evaluation (1-100):"""
