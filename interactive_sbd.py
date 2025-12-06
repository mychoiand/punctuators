
import sys
from punctuators.models import SBDModelONNX

def main():
    print("Initializing SBD Model... Please wait.", flush=True)
    try:
        # Load the model (downloads automatically if needed)
        model = SBDModelONNX.from_pretrained("sbd_multi_lang")
        print("Model loaded successfully!", flush=True)
        print("-" * 50, flush=True)
        print("Enter text to segment (type 'q' or 'exit' to quit):", flush=True)
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return

    while True:
        try:
            # Get input from user
            user_input = input("\nInput (q to quit) > ").strip()

            # Check for exit condition
            if user_input.lower() in ('q', 'exit', 'quit'):
                print("Exiting...", flush=True)
                break

            if not user_input:
                continue

            # Run inference
            # infer expects a list of strings
            results = model.infer([user_input])
            segments = results[0] # We only sent one input

            print("Output Segments:", flush=True)
            for i, segment in enumerate(segments, 1):
                print(f"  {i}: {segment}", flush=True)

        except KeyboardInterrupt:
            print("\nExiting...", flush=True)
            break
        except Exception as e:
            print(f"An error occurred during inference: {e}", flush=True)

if __name__ == "__main__":
    main()
