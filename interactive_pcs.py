
import sys
from punctuators.models import PunctCapSegModelONNX

def main():
    print("Initializing PCS Model (pcs_47lang)... Please wait.", flush=True)
    try:
        model = PunctCapSegModelONNX.from_pretrained("pcs_47lang")
        print("Model loaded successfully!", flush=True)
        print("-" * 50, flush=True)
        print("This model restores punctuation, capitalization, AND segments sentences.", flush=True)
        print("Enter raw text (type 'q' or 'exit' to quit):", flush=True)
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return

    while True:
        try:
            user_input = input("\nInput (q to quit) > ").strip()

            if user_input.lower() in ('q', 'exit', 'quit'):
                print("Exiting...", flush=True)
                break

            if not user_input:
                continue

            results = model.infer([user_input])
            output_data = results[0]

            print("Output:", flush=True)
            if isinstance(output_data, list):
                for line in output_data:
                    print(f"  {line}", flush=True)
            else:
                print(f"  {output_data}", flush=True)

        except KeyboardInterrupt:
            print("\nExiting...", flush=True)
            break
        except Exception as e:
            print(f"An error occurred during inference: {e}", flush=True)

if __name__ == "__main__":
    main()
