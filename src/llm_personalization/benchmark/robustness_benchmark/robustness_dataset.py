from llm_personalization.benchmark.personalization_system import PersonalizationItem, PersonalizationDataset
from datasets import load_dataset
from dataclasses import dataclass
import re
import string


@dataclass
class RobustnessQuestion:
    question_id: str
    question_text: str
    options: list[str]
    option_letters: list[str]
    correct_letter: str
    source: str  # "mmlu_pro", "truthfulqa", or "bbq"
    metadata: dict | None = None


def load_robustness_questions(
    include_mmlu_pro: bool = True,
    include_truthfulqa: bool = True,
    include_bbq: bool = False,
    mmlu_pro_limit: int | None = None,
    truthfulqa_limit: int | None = None,
    bbq_limit: int | None = None,
    seed: int = 42,
) -> list[RobustnessQuestion]:
    questions = []

    if include_mmlu_pro:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        if mmlu_pro_limit is not None:
            ds = ds.shuffle(seed=seed).select(range(min(mmlu_pro_limit, len(ds))))
        for row in ds:
            option_letters = list(string.ascii_uppercase[:len(row["options"])])
            questions.append(RobustnessQuestion(
                question_id=f"mmlu_pro_{row['question_id']}",
                question_text=row["question"],
                options=row["options"],
                option_letters=option_letters,
                correct_letter=row["answer"],
                source="mmlu_pro",
            ))

    if include_truthfulqa:
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
        if truthfulqa_limit is not None:
            ds = ds.shuffle(seed=seed).select(range(min(truthfulqa_limit, len(ds))))
        for idx, row in enumerate(ds):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            option_letters = list(string.ascii_uppercase[:len(choices)])
            correct_idx = labels.index(1)
            questions.append(RobustnessQuestion(
                question_id=f"truthfulqa_{idx}",
                question_text=row["question"],
                options=choices,
                option_letters=option_letters,
                correct_letter=option_letters[correct_idx],
                source="truthfulqa",
            ))

    if include_bbq:
        ds = load_dataset("heegyu/bbq", split="test", trust_remote_code=True)
        # Only use disambiguated examples (unambiguous correct answer)
        ds = ds.filter(lambda row: row["context_condition"] == "disambig")
        if bbq_limit is not None:
            ds = ds.shuffle(seed=seed).select(range(min(bbq_limit, len(ds))))
        for row in ds:
            options = [row["ans0"], row["ans1"], row["ans2"]]
            option_letters = ["A", "B", "C"]
            correct_letter = option_letters[row["label"]]
            questions.append(RobustnessQuestion(
                question_id=f"bbq_{row['example_id']}",
                question_text=f"{row['context']}\n{row['question']}",
                options=options,
                option_letters=option_letters,
                correct_letter=correct_letter,
                source="bbq",
                metadata={"category": row["category"], "question_polarity": row["question_polarity"]},
            ))

    return questions


def format_mc_prompt(question: RobustnessQuestion) -> str:
    prompt = f"{question.question_text}\n\n"
    for letter, option in zip(question.option_letters, question.options):
        prompt += f"{letter}. {option}\n"
    prompt += "\nAnswer with just the letter of the correct answer."
    return prompt


def parse_answer_letter(response: str) -> str | None:
    response = response.strip()
    # Try: first character is a letter
    if response and response[0] in string.ascii_uppercase:
        return response[0]
    # Try: find a standalone letter
    match = re.search(r'\b([A-J])\b', response)
    if match:
        return match.group(1)
    return None


class RobustnessDataset(PersonalizationDataset):
    """Wraps an existing PersonalizationDataset, replacing current_messages with benchmark questions.

    One item per question. Each question is paired with a different user
    (cycling through users), so each question gets one user's history.
    """
    def __init__(
        self,
        base_dataset: PersonalizationDataset,
        questions: list[RobustnessQuestion],
    ):
        self.base_dataset = base_dataset
        self.questions = questions

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> PersonalizationItem:
        user_idx = index % len(self.base_dataset)
        base_item = self.base_dataset[user_idx]
        question = self.questions[index]
        return PersonalizationItem(
            user_id=base_item.user_id,
            conversation_history=base_item.conversation_history,
            current_messages=[{"role": "user", "content": format_mc_prompt(question)}],
        )

    def get_question(self, index: int) -> RobustnessQuestion:
        return self.questions[index]
