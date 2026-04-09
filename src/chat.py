from src.embeddings import LocalEmbeddings
from src.qa import QASystem

class ChatInterface:
    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.qa_system = QASystem(self.embeddings)
    
    def process_query(self, user_query: str) -> str:
        """Process user query and return answer."""
        return self.qa_system.answer_question(user_query)