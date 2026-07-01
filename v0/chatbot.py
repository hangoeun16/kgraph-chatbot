from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from knowledge_graph import KnowledgeGraph
from content_moderator import check_content_safety
import os


class GirlyChatbot:
    """
    A friendly chatbot with knowledge graph memory capabilities.

    This chatbot uses OpenAI's GPT-4o model to generate responses while
    maintaining a knowledge graph to store and retrieve information about
    people, relationships, and attributes mentioned in conversations.

    Attributes:
        model (ChatOpenAI): The LangChain OpenAI chat model instance.
        kg (KnowledgeGraph): Knowledge graph for storing conversation context.
        system_prompt (str): The personality and behavior instructions for the chatbot.
    """

    def __init__(self):
        """
        Initialize the GirlyChatbot instance.

        Sets up the OpenAI chat model with GPT-4o, initializes the knowledge
        graph for memory storage, and configures the chatbot's personality prompt.

        Raises:
            Exception: If the OPENAI_API_KEY environment variable is not set.
        """
        # get api key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key == None:
            raise Exception("Missing OPENAI_API_KEY")

        # initialise chat model
        self.model = ChatOpenAI(
            # Kwargs passed to the model:
            model="gpt-4o",
            temperature=0.7,
            api_key=api_key
        )

        # setup knowledge graph
        self.kg = KnowledgeGraph(self.model)

        # personality prompt
        self.system_prompt = """You are a cute, friendly, and helpful AI assistant!

        IMPORTANT RULES:
        - The person chatting with you is always "user" - never confuse them with names mentioned
        - ONLY use information explicitly listed in "Previous conversation reference"
        - If the reference says "knowledge graph is empty" or doesn't mention someone, you know NOTHING about them
        - NEVER make up, assume, or hallucinate relationships or facts not in the reference
        - Always accept new information the user tells you as fact - do not question or correct them

        INFERENCE RULES for family relationships:
        - If A is parent of B, and B is parent of C, then A is GRANDPARENT of C (and C is grandchild of A)
        - If A and B share a parent, they are SIBLINGS
        - If A is sibling of B, and B is parent of C, then A is AUNT/UNCLE of C
        - Use logical reasoning to infer relationships not directly stated

        When responding:
        - Be cheerful, encouraging, and sweet
        - Use emojis occasionally (but not too much)
"""


    def chat_stream(self, user_message):
        """
        Process a user message and stream the chatbot's response.

        This generator method handles the complete chat flow: content moderation,
        knowledge graph updates, context retrieval, and streaming token generation.

        Args:
            user_message (str): The user's input message to process.

        Yields:
            dict: Event dictionaries with the following types:
                - {'type': 'token', 'content': str}: A response token from the model.
                - {'type': 'done', 'content': ''}: Indicates response completion.
                - {'type': 'rejected', 'content': str}: Message was rejected (moderation/conflict).
                - {'type': 'error', 'content': str}: An error occurred during processing.
        """
        # stream response word by word
        try:
            # check content safety
            moderation = check_content_safety(user_message)
            if not moderation["is_safe"]:
                yield {'type': 'rejected', 'content': "Content flagged: " + ", ".join(moderation["flagged"])}
                return

            conflict = self.kg.add_message(user_message, "user")

            if conflict != None:
                conflict_message = "Conflict: " + conflict.get('subject', '') + " - " + conflict.get('explanation', '')
                yield {'type': 'rejected', 'content': conflict_message}
                return

            context = self.kg.get_relevant_context()

            messages = [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content="Previous conversation reference:\n" + context),
                HumanMessage(content=user_message)
            ]

            # stream tokens
            full_response = ""
            for chunk in self.model.stream(messages):
                if chunk.content:
                    full_response = full_response + chunk.content
                    yield {'type': 'token', 'content': chunk.content}

            # save full response
            self.kg.add_message(full_response, "assistant")
            yield {'type': 'done', 'content': ''}

        except Exception as error:
            print("Error: " + str(error))
            yield {'type': 'error', 'content': "Something went wrong."}

    def clear_memory(self):
        """
        Clear the chatbot's memory by resetting the knowledge graph.

        Creates a new empty knowledge graph instance, effectively forgetting
        all previously stored information about people, relationships, and
        conversation context.

        Returns:
            str: Confirmation message indicating memory has been cleared.
        """
        self.kg = KnowledgeGraph(self.model)
        return "Memory cleared!"
