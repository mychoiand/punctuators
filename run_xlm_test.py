
from typing import List
from punctuators.models import PunctCapSegModelONNX

def run_test():
    print("Loading XLM-Roberta PCS Model (Heavy/High Accuracy)...")
    repo_id = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    try:
        # Pass the full HF Repo ID to load from that specific repo
        m = PunctCapSegModelONNX.from_pretrained(repo_id)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model loaded successfully.")

    input_texts: List[str] = [
        "hello world how are you today i am fine thank you",
        "안녕하세요 저는 딥라고 합니다 오늘 날씨가 참 좋네요",
        "this implies that the agent is not working correctly however we can fix it",
    ]

    print("Running inference (this might be slower)...")
    try:
        results = m.infer(input_texts)
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    for input_text, output in zip(input_texts, results):
        print(f"Input: {input_text}")
        print(f"Output: {output}")
        print("-" * 20)

if __name__ == "__main__":
    run_test()
