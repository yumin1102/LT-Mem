import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ltmem.memory_qa import answer_memory_question


def main() -> None:
    import argparse
    from ltmem.model_runtime import ModelRuntime

    parser = argparse.ArgumentParser(
        description="Ask natural-language questions about saved LT-Mem memory", allow_abbrev=False
    )
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("gemini", "qwen"),
        help="Default: Gemini for natural answers, Qwen for structured events",
    )
    parser.add_argument(
        "--question", help="Omit for an interactive question loop; each question is independent"
    )
    parser.add_argument("--mode", choices=("natural", "events"), default="natural")
    parser.add_argument(
        "--precision",
        choices=("fp16", "4bit"),
        default="fp16",
        help="Local Qwen QA precision (default: fp16)",
    )
    args = parser.parse_args()
    provider = args.provider or ("gemini" if args.mode == "natural" else "qwen")
    context = json.loads(args.memory.read_text(encoding="utf-8"))
    if provider == "qwen":
        import os

        os.environ["USE_4BIT"] = "true" if args.precision == "4bit" else "false"
    runtime = ModelRuntime(provider_kind=provider)
    retriever = None
    if args.mode == "natural":
        from ltmem.memory_retrieval import MemoryRetriever

        retriever = MemoryRetriever(context)

    def answer(question):
        if args.mode == "events":
            from ltmem.event_queries import answer_event_question

            result = answer_event_question(question, context, model_runtime=runtime)
            return json.dumps({"answer": result["answer"]}, ensure_ascii=False)
        return answer_memory_question(question, context, model_runtime=runtime, retriever=retriever)

    if args.question:
        print(answer(args.question))
        return
    print("Ask about this memory. Enter /exit to finish. Each question is independent.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/exit":
            break
        if question:
            print(answer(question))


if __name__ == "__main__":
    main()
