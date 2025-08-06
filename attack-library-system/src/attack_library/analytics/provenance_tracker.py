"""Provenance tracking system for attack sources and modifications."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from ..core.models import Attack

logger = logging.getLogger(__name__)


class ProvenanceEventType(Enum):
    """Types of provenance events."""

    CREATION = "creation"
    MODIFICATION = "modification"
    DERIVATION = "derivation"
    VERIFICATION = "verification"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ANNOTATION = "annotation"
    DELETION = "deletion"
    RESTORATION = "restoration"


@dataclass
class ProvenanceAgent:
    """Entity responsible for a provenance event."""

    id: str
    name: str
    type: str  # 'human', 'system', 'ai', 'tool'
    contact_info: Optional[str] = None
    organization: Optional[str] = None
    version: Optional[str] = None  # For tools/systems

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceAgent":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProvenanceEvent:
    """Individual provenance event record."""

    id: str
    attack_id: str
    event_type: ProvenanceEventType
    timestamp: datetime
    agent: ProvenanceAgent
    description: str

    # Event-specific data
    changes: Optional[Dict[str, Any]] = None  # What changed
    source_data: Optional[Dict[str, Any]] = None  # Source of derivation
    validation_results: Optional[Dict[str, Any]] = None  # Validation outcomes
    metadata: Optional[Dict[str, Any]] = None  # Additional event metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        data["agent"] = self.agent.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceEvent":
        """Create from dictionary."""
        data["event_type"] = ProvenanceEventType(data["event_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["agent"] = ProvenanceAgent.from_dict(data["agent"])
        return cls(**data)


@dataclass
class AttackProvenance:
    """Complete provenance record for an attack."""

    attack_id: str
    creation_event: ProvenanceEvent
    events: List[ProvenanceEvent]
    lineage: List[str]  # Chain of parent attack IDs

    # Derived information
    total_modifications: int = 0
    last_modified: Optional[datetime] = None
    contributors: Set[str] = None  # Set of contributor IDs
    source_chain: List[str] = None  # Chain of sources

    def __post_init__(self):
        """Calculate derived information."""
        if self.contributors is None:
            self.contributors = set()
        if self.source_chain is None:
            self.source_chain = []

        # Calculate statistics
        self.total_modifications = len(
            [e for e in self.events if e.event_type == ProvenanceEventType.MODIFICATION]
        )

        # Find last modification
        modification_events = [
            e
            for e in self.events
            if e.event_type
            in [ProvenanceEventType.MODIFICATION, ProvenanceEventType.TRANSFORMATION]
        ]
        if modification_events:
            self.last_modified = max(e.timestamp for e in modification_events)

        # Collect contributors
        self.contributors.add(self.creation_event.agent.id)
        for event in self.events:
            self.contributors.add(event.agent.id)

        # Build source chain
        self._build_source_chain()

    def _build_source_chain(self):
        """Build the chain of sources from events."""
        sources = []

        # Start with creation source
        if self.creation_event.source_data:
            original_source = self.creation_event.source_data.get("original_source")
            if original_source:
                sources.append(original_source)

        # Add derivation sources
        for event in self.events:
            if event.event_type == ProvenanceEventType.DERIVATION and event.source_data:
                source_attack = event.source_data.get("source_attack_id")
                if source_attack:
                    sources.append(f"attack:{source_attack}")

        self.source_chain = sources

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attack_id": self.attack_id,
            "creation_event": self.creation_event.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "lineage": self.lineage,
            "total_modifications": self.total_modifications,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "contributors": list(self.contributors),
            "source_chain": self.source_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackProvenance":
        """Create from dictionary."""
        creation_event = ProvenanceEvent.from_dict(data["creation_event"])
        events = [ProvenanceEvent.from_dict(event_data) for event_data in data["events"]]

        provenance = cls(
            attack_id=data["attack_id"],
            creation_event=creation_event,
            events=events,
            lineage=data["lineage"],
        )

        # Restore derived data
        if data.get("contributors"):
            provenance.contributors = set(data["contributors"])
        if data.get("source_chain"):
            provenance.source_chain = data["source_chain"]
        if data.get("last_modified"):
            provenance.last_modified = datetime.fromisoformat(data["last_modified"])

        return provenance


class ProvenanceTracker:
    """System for tracking attack provenance and lineage."""

    def __init__(self, data_file: Optional[Path] = None):
        """
        Initialize provenance tracker.

        Args:
            data_file: File to store provenance data
        """
        self.data_file = data_file or Path("attack_provenance.json")
        self.provenance_records: Dict[str, AttackProvenance] = {}
        self.agents: Dict[str, ProvenanceAgent] = {}

        # Load existing data
        self.load_data()

        # Register default system agent
        self._register_default_agents()

    def _register_default_agents(self):
        """Register default system agents."""
        system_agent = ProvenanceAgent(
            id="system", name="Attack Library System", type="system", version="1.0"
        )
        self.agents[system_agent.id] = system_agent

    def load_data(self):
        """Load provenance data from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r") as f:
                    data = json.load(f)

                    # Load agents
                    for agent_data in data.get("agents", []):
                        agent = ProvenanceAgent.from_dict(agent_data)
                        self.agents[agent.id] = agent

                    # Load provenance records
                    for record_data in data.get("provenance_records", []):
                        record = AttackProvenance.from_dict(record_data)
                        self.provenance_records[record.attack_id] = record

                logger.info(
                    f"Loaded {len(self.provenance_records)} provenance records and {len(self.agents)} agents"
                )
            except Exception as e:
                logger.error(f"Failed to load provenance data: {e}")

    def save_data(self):
        """Save provenance data to file."""
        try:
            data = {
                "metadata": {
                    "total_records": len(self.provenance_records),
                    "total_agents": len(self.agents),
                    "last_updated": datetime.now().isoformat(),
                    "data_version": "1.0",
                },
                "agents": [agent.to_dict() for agent in self.agents.values()],
                "provenance_records": [
                    record.to_dict() for record in self.provenance_records.values()
                ],
            }

            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.provenance_records)} provenance records")
        except Exception as e:
            logger.error(f"Failed to save provenance data: {e}")

    def register_agent(
        self,
        name: str,
        agent_type: str,
        agent_id: Optional[str] = None,
        contact_info: Optional[str] = None,
        organization: Optional[str] = None,
        version: Optional[str] = None,
    ) -> ProvenanceAgent:
        """
        Register a new provenance agent.

        Args:
            name: Agent name
            agent_type: Type of agent ('human', 'system', 'ai', 'tool')
            agent_id: Optional specific ID
            contact_info: Contact information
            organization: Organization
            version: Version for tools/systems

        Returns:
            Created agent
        """
        if agent_id is None:
            agent_id = f"{agent_type}_{name.lower().replace(' ', '_')}"

        agent = ProvenanceAgent(
            id=agent_id,
            name=name,
            type=agent_type,
            contact_info=contact_info,
            organization=organization,
            version=version,
        )

        self.agents[agent.id] = agent
        self.save_data()

        logger.info(f"Registered agent: {agent.name} ({agent.id})")
        return agent

    def record_attack_creation(
        self,
        attack: Attack,
        agent_id: str,
        description: str,
        source_data: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEvent:
        """
        Record the creation of an attack.

        Args:
            attack: Created attack
            agent_id: ID of creating agent
            description: Description of creation
            source_data: Optional source information

        Returns:
            Created provenance event
        """
        if agent_id not in self.agents:
            logger.warning(f"Unknown agent ID: {agent_id}, using system agent")
            agent_id = "system"

        # Create creation event
        creation_event = ProvenanceEvent(
            id=str(uuid4())[:8],
            attack_id=attack.id,
            event_type=ProvenanceEventType.CREATION,
            timestamp=datetime.now(),
            agent=self.agents[agent_id],
            description=description,
            source_data=source_data,
            metadata={
                "attack_title": attack.title,
                "attack_category": attack.category.value,
                "attack_severity": attack.severity.value,
                "content_length": len(attack.content),
                "target_models": attack.target_models,
            },
        )

        # Create provenance record
        lineage = []
        if attack.parent_id:
            lineage.append(attack.parent_id)
            # Inherit parent lineage if available
            if attack.parent_id in self.provenance_records:
                parent_lineage = self.provenance_records[attack.parent_id].lineage
                lineage.extend(parent_lineage)

        provenance = AttackProvenance(
            attack_id=attack.id, creation_event=creation_event, events=[], lineage=lineage
        )

        self.provenance_records[attack.id] = provenance
        self.save_data()

        logger.info(f"Recorded creation of attack {attack.id} by {agent_id}")
        return creation_event

    def record_attack_modification(
        self,
        attack_id: str,
        agent_id: str,
        description: str,
        changes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEvent:
        """
        Record modification of an attack.

        Args:
            attack_id: ID of modified attack
            agent_id: ID of modifying agent
            description: Description of modification
            changes: Dictionary of changes made
            metadata: Optional additional metadata

        Returns:
            Created provenance event
        """
        if agent_id not in self.agents:
            logger.warning(f"Unknown agent ID: {agent_id}, using system agent")
            agent_id = "system"

        event = ProvenanceEvent(
            id=str(uuid4())[:8],
            attack_id=attack_id,
            event_type=ProvenanceEventType.MODIFICATION,
            timestamp=datetime.now(),
            agent=self.agents[agent_id],
            description=description,
            changes=changes,
            metadata=metadata or {},
        )

        # Add to provenance record
        if attack_id in self.provenance_records:
            self.provenance_records[attack_id].events.append(event)
            # Update derived info
            self.provenance_records[attack_id].__post_init__()
        else:
            logger.warning(f"No provenance record found for attack {attack_id}")

        self.save_data()

        logger.info(f"Recorded modification of attack {attack_id} by {agent_id}")
        return event

    def record_attack_derivation(
        self,
        new_attack_id: str,
        source_attack_id: str,
        agent_id: str,
        description: str,
        derivation_method: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEvent:
        """
        Record derivation of one attack from another.

        Args:
            new_attack_id: ID of derived attack
            source_attack_id: ID of source attack
            agent_id: ID of agent performing derivation
            description: Description of derivation
            derivation_method: Method used for derivation
            metadata: Optional additional metadata

        Returns:
            Created provenance event
        """
        if agent_id not in self.agents:
            logger.warning(f"Unknown agent ID: {agent_id}, using system agent")
            agent_id = "system"

        event = ProvenanceEvent(
            id=str(uuid4())[:8],
            attack_id=new_attack_id,
            event_type=ProvenanceEventType.DERIVATION,
            timestamp=datetime.now(),
            agent=self.agents[agent_id],
            description=description,
            source_data={
                "source_attack_id": source_attack_id,
                "derivation_method": derivation_method,
            },
            metadata=metadata or {},
        )

        # Add to provenance record
        if new_attack_id in self.provenance_records:
            self.provenance_records[new_attack_id].events.append(event)
            self.provenance_records[new_attack_id].__post_init__()

        self.save_data()

        logger.info(f"Recorded derivation of attack {new_attack_id} from {source_attack_id}")
        return event

    def record_attack_verification(
        self,
        attack_id: str,
        agent_id: str,
        verification_result: bool,
        description: str,
        validation_results: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEvent:
        """
        Record verification of an attack.

        Args:
            attack_id: ID of verified attack
            agent_id: ID of verifying agent
            verification_result: Result of verification
            description: Description of verification
            validation_results: Optional validation results

        Returns:
            Created provenance event
        """
        if agent_id not in self.agents:
            logger.warning(f"Unknown agent ID: {agent_id}, using system agent")
            agent_id = "system"

        event = ProvenanceEvent(
            id=str(uuid4())[:8],
            attack_id=attack_id,
            event_type=ProvenanceEventType.VERIFICATION,
            timestamp=datetime.now(),
            agent=self.agents[agent_id],
            description=description,
            validation_results={
                "verified": verification_result,
                "details": validation_results or {},
            },
        )

        # Add to provenance record
        if attack_id in self.provenance_records:
            self.provenance_records[attack_id].events.append(event)
            self.provenance_records[attack_id].__post_init__()

        self.save_data()

        logger.info(f"Recorded verification of attack {attack_id}: {verification_result}")
        return event

    def get_attack_provenance(self, attack_id: str) -> Optional[AttackProvenance]:
        """Get complete provenance record for an attack."""
        return self.provenance_records.get(attack_id)

    def get_attack_lineage(self, attack_id: str) -> List[str]:
        """Get the complete lineage (ancestry) of an attack."""
        if attack_id not in self.provenance_records:
            return []

        return self.provenance_records[attack_id].lineage

    def get_attack_descendants(self, attack_id: str) -> List[str]:
        """Get all attacks derived from this attack."""
        descendants = []

        for provenance in self.provenance_records.values():
            if attack_id in provenance.lineage:
                descendants.append(provenance.attack_id)

        return descendants

    def get_attack_contributors(self, attack_id: str) -> List[ProvenanceAgent]:
        """Get all agents who contributed to an attack."""
        if attack_id not in self.provenance_records:
            return []

        contributor_ids = self.provenance_records[attack_id].contributors
        return [self.agents[agent_id] for agent_id in contributor_ids if agent_id in self.agents]

    def get_agent_contributions(self, agent_id: str) -> List[str]:
        """Get all attacks an agent contributed to."""
        contributions = []

        for provenance in self.provenance_records.values():
            if agent_id in provenance.contributors:
                contributions.append(provenance.attack_id)

        return contributions

    def generate_provenance_report(self, attack_id: str) -> Dict[str, Any]:
        """Generate comprehensive provenance report for an attack."""
        if attack_id not in self.provenance_records:
            return {"error": f"No provenance record found for attack {attack_id}"}

        provenance = self.provenance_records[attack_id]

        # Build detailed timeline
        timeline = [provenance.creation_event]
        timeline.extend(provenance.events)
        timeline.sort(key=lambda e: e.timestamp)

        # Analyze contribution patterns
        contributor_analysis = {}
        for contributor_id in provenance.contributors:
            if contributor_id in self.agents:
                agent = self.agents[contributor_id]
                events_by_agent = [e for e in timeline if e.agent.id == contributor_id]

                contributor_analysis[contributor_id] = {
                    "agent_info": agent.to_dict(),
                    "total_events": len(events_by_agent),
                    "event_types": Counter(e.event_type.value for e in events_by_agent),
                    "first_contribution": min(e.timestamp for e in events_by_agent).isoformat(),
                    "last_contribution": max(e.timestamp for e in events_by_agent).isoformat(),
                }

        # Impact analysis
        descendants = self.get_attack_descendants(attack_id)
        lineage = self.get_attack_lineage(attack_id)

        report = {
            "attack_id": attack_id,
            "provenance_summary": {
                "total_events": len(timeline),
                "total_contributors": len(provenance.contributors),
                "total_modifications": provenance.total_modifications,
                "creation_date": provenance.creation_event.timestamp.isoformat(),
                "last_modified": provenance.last_modified.isoformat()
                if provenance.last_modified
                else None,
                "lineage_depth": len(lineage),
                "descendant_count": len(descendants),
            },
            "timeline": [
                {
                    "event_id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "agent": event.agent.name,
                    "description": event.description,
                }
                for event in timeline
            ],
            "contributor_analysis": contributor_analysis,
            "lineage": {
                "ancestors": lineage,
                "descendants": descendants,
                "source_chain": provenance.source_chain,
            },
            "quality_indicators": {
                "is_well_documented": len(timeline) >= 3,
                "has_verification": any(
                    e.event_type == ProvenanceEventType.VERIFICATION for e in timeline
                ),
                "multi_contributor": len(provenance.contributors) > 1,
                "has_derivation_source": any(
                    e.event_type == ProvenanceEventType.DERIVATION for e in timeline
                ),
            },
        }

        return report

    def export_provenance_data(
        self, output_path: Path, format_type: str = "json", attack_ids: Optional[List[str]] = None
    ):
        """
        Export provenance data.

        Args:
            output_path: Output file path
            format_type: Export format ('json', 'csv')
            attack_ids: Optional list of specific attacks to export
        """
        if attack_ids:
            records_to_export = {
                aid: record for aid, record in self.provenance_records.items() if aid in attack_ids
            }
        else:
            records_to_export = self.provenance_records

        if format_type.lower() == "json":
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_records": len(records_to_export),
                    "total_agents": len(self.agents),
                },
                "agents": [agent.to_dict() for agent in self.agents.values()],
                "provenance_records": [record.to_dict() for record in records_to_export.values()],
            }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format_type.lower() == "csv":
            import csv

            # Flatten provenance data for CSV
            with open(output_path, "w", newline="") as f:
                fieldnames = [
                    "attack_id",
                    "event_type",
                    "timestamp",
                    "agent_name",
                    "agent_type",
                    "description",
                    "total_modifications",
                    "contributor_count",
                    "lineage_depth",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for record in records_to_export.values():
                    # Write creation event
                    writer.writerow(
                        {
                            "attack_id": record.attack_id,
                            "event_type": record.creation_event.event_type.value,
                            "timestamp": record.creation_event.timestamp.isoformat(),
                            "agent_name": record.creation_event.agent.name,
                            "agent_type": record.creation_event.agent.type,
                            "description": record.creation_event.description,
                            "total_modifications": record.total_modifications,
                            "contributor_count": len(record.contributors),
                            "lineage_depth": len(record.lineage),
                        }
                    )

                    # Write other events
                    for event in record.events:
                        writer.writerow(
                            {
                                "attack_id": record.attack_id,
                                "event_type": event.event_type.value,
                                "timestamp": event.timestamp.isoformat(),
                                "agent_name": event.agent.name,
                                "agent_type": event.agent.type,
                                "description": event.description,
                                "total_modifications": "",
                                "contributor_count": "",
                                "lineage_depth": "",
                            }
                        )

        logger.info(f"Exported {len(records_to_export)} provenance records to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance tracking statistics."""
        if not self.provenance_records:
            return {"total_records": 0, "total_agents": len(self.agents), "total_events": 0}

        total_events = sum(
            len(record.events) + 1  # +1 for creation event
            for record in self.provenance_records.values()
        )

        # Event type distribution
        event_types = []
        for record in self.provenance_records.values():
            event_types.append(record.creation_event.event_type)
            event_types.extend(event.event_type for event in record.events)

        event_type_dist = Counter(et.value for et in event_types)

        # Agent type distribution
        agent_type_dist = Counter(agent.type for agent in self.agents.values())

        # Modification statistics
        modifications_per_attack = [
            record.total_modifications for record in self.provenance_records.values()
        ]

        # Lineage statistics
        lineage_depths = [len(record.lineage) for record in self.provenance_records.values()]

        return {
            "total_records": len(self.provenance_records),
            "total_agents": len(self.agents),
            "total_events": total_events,
            "event_type_distribution": dict(event_type_dist),
            "agent_type_distribution": dict(agent_type_dist),
            "modification_statistics": {
                "average_modifications_per_attack": sum(modifications_per_attack)
                / len(modifications_per_attack)
                if modifications_per_attack
                else 0,
                "max_modifications": max(modifications_per_attack)
                if modifications_per_attack
                else 0,
                "attacks_with_modifications": len([m for m in modifications_per_attack if m > 0]),
            },
            "lineage_statistics": {
                "average_lineage_depth": sum(lineage_depths) / len(lineage_depths)
                if lineage_depths
                else 0,
                "max_lineage_depth": max(lineage_depths) if lineage_depths else 0,
                "attacks_with_lineage": len([d for d in lineage_depths if d > 0]),
            },
        }
