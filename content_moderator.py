from openai import OpenAI

client = OpenAI()

def check_content_safety(user_text):
    """
    Check if user input contains inappropriate or harmful content.

    Uses OpenAI's Moderation API to analyze text for policy violations
    including sexual content, hate speech, harassment, self-harm, and violence.

    Args:
        user_text (str): The user input text to analyze for safety.

    Returns:
        dict: A dictionary containing:
            - 'is_safe' (bool): True if content passes moderation, False otherwise.
            - 'flagged' (list): List of category names that were flagged (e.g.,
              'sexual content', 'hate speech', 'harassment', 'self-harm',
              'sexual content involving minors', 'violence').

    Note:
        If the moderation API call fails, the function returns is_safe=True
        to allow the conversation to continue rather than blocking on errors.
    """
    # check if user input contains inappropriate content

    try:
        response = client.moderations.create(input=user_text)
        result = response.results[0]

        if result.flagged:
            flagged = []
            categories = result.categories

            if categories.sexual:
                flagged.append("sexual content")
            if categories.hate:
                flagged.append("hate speech")
            if categories.harassment:
                flagged.append("harassment")
            if categories.self_harm:
                flagged.append("self-harm")
            if categories.sexual_minors:
                flagged.append("sexual content involving minors")
            if categories.violence:
                flagged.append("violence")

            return {"is_safe": False, "flagged": flagged}

        return {"is_safe": True, "flagged": []}

    except Exception as error:
        print("Moderation error: " + str(error))
        return {"is_safe": True, "flagged": []}
