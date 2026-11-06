from ragapp.llm.base import Message, ToolSpec

SUPPORTED_CATEGORIES = [
    "Setup & App",
    "Charging & Battery",
    "Cleaning & Operation",
    "Maintenance",
    "Mopping",
    "Safety",
    "Troubleshooting",
]

SUPPORTED_MODELS = [
    "Q7", "Q7 Max", "Q7t", "Q7 Series",
    "Q10 Series",
    "Q Revo",
    "QRevo Curv",
    "QRevo Curv2 Flow",
]

_CLARIFIER_SYSTEM_PROMPT = (
    "You classify incoming questions for a Roborock robot vacuum support chatbot.\n\n"
    f"Supported models: {', '.join(SUPPORTED_MODELS)}.\n"
    f"Supported question categories: {', '.join(SUPPORTED_CATEGORIES)}.\n\n"
    "Decide whether the question (considering conversation history) has enough context to answer well.\n\n"
    "Rules:\n"
    "1. MODEL — is a specific Roborock model identifiable from the question or conversation history?\n"
    "   - If the same model appears anywhere in the history, treat it as identified for this turn.\n"
    "   - Questions that apply to all models equally (e.g. general app questions, safety) → model is NOT missing.\n"
    "2. CATEGORY — does the question map to one of the supported categories above?\n"
    "   - Very short or vague questions ('it's not working', 'help') may need category clarification.\n"
    "   - If the question is clearly out of scope, do NOT ask for clarification — the main system will refuse it.\n\n"
    "Call the classify_question tool with your determination."
)

_CLASSIFY_TOOL = ToolSpec(
    name="classify_question",
    description="Classify whether the user's question needs clarification before it can be answered.",
    input_schema={
        "type": "object",
        "properties": {
            "needs_clarification": {
                "type": "boolean",
                "description": (
                    "True when the missing model or category would significantly change "
                    "the answer and asking is worth the extra round-trip."
                ),
            },
            "missing_model": {
                "type": "boolean",
                "description": "True when no specific Roborock model is identifiable and knowing it would change the answer.",
            },
            "missing_category": {
                "type": "boolean",
                "description": "True when the question does not clearly map to any supported category.",
            },
            "clarification_question": {
                "type": "string",
                "description": (
                    "A short, friendly clarifying question to ask the user. "
                    "Ask about model first, then category — combine into one question if both are missing. "
                    "Empty string when needs_clarification is false."
                ),
            },
        },
        "required": ["needs_clarification", "missing_model", "missing_category", "clarification_question"],
    },
)


def check_needs_clarification(
    question: str, history: list[Message], llm_client
) -> str | None:
    """Return a clarifying question string if model or category is ambiguous, else None.

    Fails open: any exception from the LLM call returns None so the main chat
    path proceeds uninterrupted."""
    try:
        messages: list[Message] = [
            *history,
            {"role": "user", "content": question},
        ]
        turn = llm_client.complete_with_tools(
            messages=messages,
            tools=[_CLASSIFY_TOOL],
            system_prompt=_CLARIFIER_SYSTEM_PROMPT,
        )
        if not turn.tool_calls:
            return None
        args = turn.tool_calls[0].arguments
        if args.get("needs_clarification") and args.get("clarification_question"):
            return str(args["clarification_question"])
    except Exception:
        pass
    return None
