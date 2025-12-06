
from typing import List
from punctuators.models import SBDModelONNX

def run_test():
    print("Loading SBD model... (this may trigger a download on first run)")
    # Instantiate this model
    # This will download the ONNX and SPE models.
    # To clean up, delete this model from your HF cache directory.
    try:
        m = SBDModelONNX.from_pretrained("sbd_multi_lang")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model loaded successfully.")

    input_texts: List[str] = [
        # English (with a lot of acronyms)
        "the new d.n.a. sample has been multiplexed, and the gametes are already dividing. let's get the c.p.d. over there. dinner's at 630 p.m. see that piece on you in the l.a. times? chicago p.d. will eat him alive.",
        # Korean example (adding one for relevance as user seems to be Korean)
        "안녕하세요. 저는 오늘 점심으로 김치찌개를 먹었습니다. 날씨가 참 좋네요, 산책이라도 갈까요?",
    ]

    print("Running inference...")
    # Run inference
    try:
        results: List[List[str]] = m.infer(input_texts)
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # Print each input and it's segmented outputs
    for input_text, output_texts in zip(input_texts, results):
        print(f"Input: {input_text}")
        print(f"Outputs:")
        for text in output_texts:
            print(f"\t{text}")
        print("-" * 20)

if __name__ == "__main__":
    run_test()
