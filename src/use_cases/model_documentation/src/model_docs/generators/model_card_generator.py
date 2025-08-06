"""Generate model cards and documentation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..analyzers import MetadataExtractor, ModelInspector
from ..models import (
    EthicalConsiderations,
    ModelCard,
    ModelMetadata,
    PerformanceMetrics,
    TrainingConfig,
)
from .template_engine import TemplateEngine

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """Generate comprehensive model cards."""

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the model card generator.

        Args:
            template_dir: Directory containing templates
        """
        self.template_engine = TemplateEngine(template_dir)
        self.inspector = ModelInspector()
        self.extractor = MetadataExtractor()

    def generate_model_card(
        self,
        model: Any,
        metadata: Optional[ModelMetadata] = None,
        training_config: Optional[TrainingConfig] = None,
        performance: Optional[PerformanceMetrics] = None,
        ethical_considerations: Optional[EthicalConsiderations] = None,
        **kwargs,
    ) -> ModelCard:
        """
        Generate a complete model card.

        Args:
            model: The model to document
            metadata: Model metadata (auto-extracted if None)
            training_config: Training configuration
            performance: Performance metrics
            ethical_considerations: Ethical considerations
            **kwargs: Additional fields

        Returns:
            Complete ModelCard object
        """
        # Extract metadata if not provided
        if metadata is None:
            metadata = self.extractor.extract_metadata(model)

        # Create model card
        model_card = ModelCard(
            metadata=metadata,
            training_config=training_config,
            performance=performance,
            ethical_considerations=ethical_considerations,
            usage_guidelines=kwargs.get("usage_guidelines"),
            citation=kwargs.get("citation"),
            license=kwargs.get("license"),
            contact_info=kwargs.get("contact_info"),
            references=kwargs.get("references", []),
            changelog=kwargs.get("changelog", []),
        )

        return model_card

    def render_markdown(self, model_card: ModelCard) -> str:
        """
        Render model card as markdown.

        Args:
            model_card: The model card to render

        Returns:
            Markdown string
        """
        context = {
            "metadata": model_card.metadata,
            "training_config": model_card.training_config,
            "performance": model_card.performance,
            "ethical_considerations": model_card.ethical_considerations,
            "usage_guidelines": model_card.usage_guidelines,
            "citation": model_card.citation,
            "license": model_card.license,
            "contact_info": model_card.contact_info,
            "references": model_card.references,
            "datetime": datetime,
        }

        return self.template_engine.render("model_card.md.j2", context)

    def save_model_card(
        self, model_card: ModelCard, output_path: Path, format: str = "markdown"
    ) -> Path:
        """
        Save model card to file.

        Args:
            model_card: The model card to save
            output_path: Output file path
            format: Output format (markdown, json, pdf)

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = self.render_markdown(model_card)
            output_path = output_path.with_suffix(".md")
            with open(output_path, "w") as f:
                f.write(content)
        elif format == "json":
            import json

            output_path = output_path.with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(model_card.dict(), f, indent=2, default=str)
        elif format == "pdf":
            # Generate PDF from markdown
            markdown_content = self.render_markdown(model_card)
            output_path = self._generate_pdf(markdown_content, output_path.with_suffix(".pdf"))
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Model card saved to {output_path}")
        return output_path

    def _generate_pdf(self, markdown_content: str, output_path: Path) -> Path:
        """Generate PDF from markdown content."""
        try:
            import markdown
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

            # Convert markdown to HTML
            html = markdown.markdown(markdown_content)

            # Create PDF
            doc = SimpleDocTemplate(str(output_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Parse HTML and add to story
            lines = html.split("\n")
            for line in lines:
                if line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                    story.append(Spacer(1, 12))

            doc.build(story)
            return output_path

        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise
