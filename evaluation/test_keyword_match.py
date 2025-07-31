"""Test script for keyword match evaluator."""

from keyword_match import keyword_match


def test_keyword_match():
    """Test the keyword match evaluator with various scenarios."""

    print("Testing keyword match evaluator...\n")

    # Test 1: Basic match
    result = keyword_match(
        "Miguel de Cervantes wrote Don Quixote.",
        ["Cervantes", "Miguel de Cervantes"]
    )
    print("Test 1 - Basic match:")
    print("  Response: 'Miguel de Cervantes wrote Don Quixote.'")
    print(f"  Result: {result}")
    print()

    # Test 2: No match
    result = keyword_match(
        "Shakespeare wrote Hamlet.",
        ["Cervantes", "Miguel de Cervantes"]
    )
    print("Test 2 - No match:")
    print("  Response: 'Shakespeare wrote Hamlet.'")
    print(f"  Result: {result}")
    print()

    # Test 3: Case insensitive match
    result = keyword_match(
        "CERVANTES was a Spanish writer.",
        ["cervantes"]
    )
    print("Test 3 - Case insensitive match:")
    print("  Response: 'CERVANTES was a Spanish writer.'")
    print(f"  Result: {result}")
    print()

    # Test 4: Word boundary test (should not match partial)
    result = keyword_match(
        "The scar on his face was visible.",
        ["car"]
    )
    print("Test 4 - Word boundary test (should not match 'car' in 'scar'):")
    print("  Response: 'The scar on his face was visible.'")
    print(f"  Result: {result}")
    print()

    # Test 5: Edge case - None response
    result = keyword_match(
        None,
        ["test"]
    )
    print("Test 5 - None response:")
    print("  Response: None")
    print(f"  Result: {result}")
    print()

    # Test 6: Edge case - Empty keywords
    result = keyword_match(
        "Some response text",
        []
    )
    print("Test 6 - Empty keywords:")
    print("  Response: 'Some response text'")
    print(f"  Result: {result}")
    print()


if __name__ == "__main__":
    test_keyword_match()
