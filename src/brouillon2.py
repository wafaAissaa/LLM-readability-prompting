import random
import re


def sample_negative_examples(text, positive_tokens):
    # Tokenize the text while preserving multi-word expressions
    tokens = []
    remaining_text = text
    num_positive = len(positive_tokens)
    random.seed(42)

    # Sort positive tokens by length in descending order to match longer phrases first
    sorted_positive_tokens = sorted(positive_tokens, key=len, reverse=True)

    for token in sorted_positive_tokens:
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        matches = pattern.finditer(remaining_text)

        for match in matches:
            start, end = match.span()
            # Add the text before the match as individual tokens
            preceding_text = remaining_text[:start].strip()
            if preceding_text:
                tokens.extend(preceding_text.split())

            # Add the matched multi-word token
            tokens.append(match.group())

            # Update the remaining text
            remaining_text = remaining_text[end:].strip()

    # Add any remaining text as individual tokens
    if remaining_text:
        tokens.extend(remaining_text.split())

    # Identify positive tokens in the text
    positive_tokens_in_text = [token for token in tokens if token.lower() in [pt.lower() for pt in positive_tokens]]

    # Split positive tokens into individual words for exclusion
    positive_words = set(word.lower() for token in positive_tokens for word in token.split())

    # Exclude positive tokens and any tokens containing positive words to get potential negative tokens
    potential_negative_tokens = [
        token for token in tokens
        if token.lower() not in [pt.lower() for pt in positive_tokens]
        and not any(word.lower() in positive_words for word in token.split())
    ]

    # Randomly sample negative tokens
    negative_tokens = random.sample(potential_negative_tokens, min(num_positive, len(potential_negative_tokens)))

    return negative_tokens

# Example usage
text = "This is an example text. This text contains positive tokens and multi-word expressions."
positive_tokens = {'positive tokens', 'multi-word expressions'}
num_positive = len(positive_tokens)

negative_tokens = sample_negative_examples(text, positive_tokens, num_positive)

print("Positive Tokens:", positive_tokens)
print("Negative Tokens:", negative_tokens)
