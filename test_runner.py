from supervisor import SupervisorAgent

def main():
    supervisor = SupervisorAgent()

    sample_text_test = input("can you add a text?")
    sample_text = input("can you add a text?")
    sample_notes = input("can you add a text?")
    audio_path = "/Users/pradyundangwal/Desktop/rajni2.ogg"  # Correct file path to your audio file
    
    print("\n--- Paraphrased Text ---")
    print(supervisor.handle_request("paraphrase", sample_text_test))

    print("\n--- Academic Rewriting ---")
    print(supervisor.handle_request("rewrite", sample_text))

    print("\n--- Summarized Notes ---")
    print(supervisor.handle_request("summarize", sample_notes))

    print("\n--- Transcribed Audio ---")
    print(supervisor.handle_request("transcribe_audio", audio_path))

if __name__ == "__main__":
    main()
