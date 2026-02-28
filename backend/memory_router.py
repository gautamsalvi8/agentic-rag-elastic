class MemoryRouter:
    def __init__(self):
        self.history = []  # list of (query, answer)

    def should_search(self, query):
        query = query.lower()
        words = query.split()

        follow_words = ["it", "that", "more", "then", "also"]

        # short follow-up → memory
        if len(words) < 4 and any(w in words for w in follow_words):
            return False

        return True  # otherwise search

    def save(self, query, answer):
        self.history.append((query, answer))

    def get_context(self, last_k=3):
        context = ""
        for q, a in self.history[-last_k:]:
            context += f"User: {q}\nAssistant: {a}\n"
        return context
