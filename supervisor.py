from paraphraser import paraphrase_text
from rewrite_academic import rewrite_to_academic_tone
from summarize_notes import summarize_notes
from audio_to_text import transcribe_audio
from note_logic import save_user_note


class SupervisorAgent:
    def __init__(self):
        pass

    def handle_request(self, task_type, data):
        if task_type == "paraphrase":
            return paraphrase_text(data)
        elif task_type == "rewrite":
            return rewrite_to_academic_tone(data)
        elif task_type == "summarize":
            return summarize_notes(data)
        elif task_type == "transcribe_audio":
            return transcribe_audio(data)
        else:
            raise ValueError("Invalid task type")