from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class ContentCategory(Enum):
    """Enumeration of content categories for classification."""
    SAFE = "safe"
    SENSITIVE = "sensitive"
    HARMFUL = "harmful"


@dataclass
class ContentConstraints:
    """Data class defining content constraints and thresholds."""
    max_length: int
    min_length: int
    allowed_categories: List[ContentCategory]
    toxicity_threshold: float
    required_disclaimers: List[str]


class ConstraintViolation(Exception):
    """Custom exception for constraint violations."""
    pass


class AIOutputConstraintPipeline:
    """Pipeline for enforcing constraints on AI-generated content."""

    def __init__(self, constraints: ContentConstraints):
        """
        Initialize the constraint pipeline with defined constraints.

        Args:
            constraints: ContentConstraints object defining the rules
        """
        self.constraints = constraints
        self._sensitive_patterns = [
            r'\b(password|secret|private|confidential)\b',
            r'\b(exploit|hack|attack)\b',
            # Add more patterns as needed
        ]

    def _check_length(self, content: str) -> bool:
        """
        Verify content length meets constraints.

        Args:
            content: The text content to check

        Returns:
            bool: True if length constraints are met
        """
        length = len(content)
        return (self.constraints.min_length <= length <= 
                self.constraints.max_length)

    def _analyze_content_category(self, content: str) -> ContentCategory:
        """
        Analyze content and determine its category.

        Args:
            content: The text content to analyze

        Returns:
            ContentCategory: The determined category of the content
        """
        # Check for sensitive patterns
        for pattern in self._sensitive_patterns:
            if re.search(pattern, content.lower()):
                return ContentCategory.SENSITIVE
        
        # Add more sophisticated content analysis here
        return ContentCategory.SAFE

    def _calculate_toxicity_score(self, content: str) -> float:
        """
        Calculate a basic toxicity score for the content.
        
        Args:
            content: The text content to analyze

        Returns:
            float: Toxicity score between 0 and 1
        """
        # This is a simplified example - in practice, you'd want to use
        # a more sophisticated toxicity detection model
        toxic_words = ['hate', 'violent', 'harmful', 'dangerous']
        score = sum(word in content.lower() for word in toxic_words) / len(toxic_words)
        return score

    def _verify_disclaimers(self, content: str) -> bool:
        """
        Verify that required disclaimers are present in the content.

        Args:
            content: The text content to check

        Returns:
            bool: True if all required disclaimers are present
        """
        return all(disclaimer.lower() in content.lower() 
                  for disclaimer in self.constraints.required_disclaimers)

    def process_output(self, content: str) -> str:
        """
        Process and validate AI-generated content against constraints.

        Args:
            content: The AI-generated content to process

        Returns:
            str: Validated and potentially modified content

        Raises:
            ConstraintViolation: If content violates defined constraints
        """
        # Check length constraints
        if not self._check_length(content):
            raise ConstraintViolation("Content length violates constraints")

        # Check content category
        category = self._analyze_content_category(content)
        if category not in self.constraints.allowed_categories:
            raise ConstraintViolation(f"Content category {category} not allowed")

        # Check toxicity
        toxicity_score = self._calculate_toxicity_score(content)
        if toxicity_score > self.constraints.toxicity_threshold:
            raise ConstraintViolation("Content exceeds toxicity threshold")

        # Verify disclaimers
        if not self._verify_disclaimers(content):
            raise ConstraintViolation("Required disclaimers missing")

        return content


# Example usage
def main():
    # Define constraints
    constraints = ContentConstraints(
        max_length=1000,
        min_length=10,
        allowed_categories=[ContentCategory.SAFE, ContentCategory.SENSITIVE],
        toxicity_threshold=0.3,
        required_disclaimers=["This is AI-generated content"]
    )

    # Initialize pipeline
    pipeline = AIOutputConstraintPipeline(constraints)

    # Test cases
    test_cases = [
        # Valid content
        """This is AI-generated content. 
        Here is some sample text that demonstrates the constraint pipeline.
        It includes the required disclaimer and maintains appropriate length.""",
        
        # Content missing disclaimer
        """Here is some content that doesn't include the required disclaimer.
        It should fail validation.""",
        
        # Content with sensitive terms
        """This is AI-generated content.
        This content includes sensitive terms like password and private information.
        It should be flagged as sensitive but still pass since sensitive content is allowed.""",
        
        # Content with toxic terms
        """This is AI-generated content.
        This content includes harmful and hate speech.
        It should fail the toxicity check.""",
        
        # Content too short
        """Too short."""
    ]

    # Process each test case
    for i, content in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("-" * 50)
        print(f"Content: {content[:100]}...")
        try:
            validated_content = pipeline.process_output(content)
            print("✓ Content validated successfully!")
        except ConstraintViolation as e:
            print(f"✗ Constraint violation: {e}")
        print("-" * 50)


if __name__ == "__main__":
    main()