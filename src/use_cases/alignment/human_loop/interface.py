"""Human-in-the-loop interface for alignment feedback and moderation."""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of human review."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ReviewPriority(Enum):
    """Priority levels for review."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewAction(Enum):
    """Actions that can be taken on content."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"


@dataclass
class ReviewRequest:
    """Request for human review."""
    
    request_id: str
    content_type: str  # "prompt", "response", "both"
    prompt: str
    response: Optional[str]
    
    # Review metadata
    priority: ReviewPriority
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Review state
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_id: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    
    # Review results
    action: Optional[ReviewAction] = None
    modified_content: Optional[str] = None
    reviewer_notes: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if review request has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "content_type": self.content_type,
            "prompt": self.prompt,
            "response": self.response,
            "priority": self.priority.value,
            "reason": self.reason,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "action": self.action.value if self.action else None,
            "modified_content": self.modified_content,
            "reviewer_notes": self.reviewer_notes
        }


@dataclass
class ReviewMetrics:
    """Metrics for review system."""
    
    total_requests: int = 0
    pending_requests: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    modified_count: int = 0
    escalated_count: int = 0
    
    average_review_time: float = 0.0
    reviewer_agreement_rate: float = 0.0
    
    by_priority: Dict[str, int] = field(default_factory=dict)
    by_reviewer: Dict[str, int] = field(default_factory=dict)


class ReviewInterface(ABC):
    """Abstract interface for human review systems."""
    
    @abstractmethod
    async def submit_for_review(self, request: ReviewRequest) -> str:
        """Submit content for review."""
        pass
        
    @abstractmethod
    async def get_review_status(self, request_id: str) -> ReviewStatus:
        """Get status of a review request."""
        pass
        
    @abstractmethod
    async def get_review_result(self, request_id: str) -> Optional[ReviewRequest]:
        """Get completed review result."""
        pass
        
    @abstractmethod
    async def get_pending_reviews(self, 
                                 reviewer_id: Optional[str] = None,
                                 priority: Optional[ReviewPriority] = None) -> List[ReviewRequest]:
        """Get pending review requests."""
        pass


class InMemoryReviewInterface(ReviewInterface):
    """In-memory implementation of review interface."""
    
    def __init__(self):
        self.reviews: Dict[str, ReviewRequest] = {}
        self.review_queue: List[str] = []
        self.metrics = ReviewMetrics()
        
    async def submit_for_review(self, request: ReviewRequest) -> str:
        """Submit content for review."""
        self.reviews[request.request_id] = request
        self.review_queue.append(request.request_id)
        
        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.pending_requests += 1
        
        priority_key = request.priority.value
        self.metrics.by_priority[priority_key] = self.metrics.by_priority.get(priority_key, 0) + 1
        
        logger.info(f"Submitted review request {request.request_id} with priority {request.priority.value}")
        
        return request.request_id
        
    async def get_review_status(self, request_id: str) -> ReviewStatus:
        """Get status of a review request."""
        if request_id in self.reviews:
            request = self.reviews[request_id]
            
            # Check if expired
            if request.is_expired() and request.status == ReviewStatus.PENDING:
                request.status = ReviewStatus.EXPIRED
                self.metrics.pending_requests -= 1
                
            return request.status
            
        return ReviewStatus.EXPIRED
        
    async def get_review_result(self, request_id: str) -> Optional[ReviewRequest]:
        """Get completed review result."""
        if request_id in self.reviews:
            request = self.reviews[request_id]
            if request.status != ReviewStatus.PENDING:
                return request
        return None
        
    async def get_pending_reviews(self,
                                 reviewer_id: Optional[str] = None,
                                 priority: Optional[ReviewPriority] = None) -> List[ReviewRequest]:
        """Get pending review requests."""
        pending = []
        
        for request_id in self.review_queue:
            if request_id in self.reviews:
                request = self.reviews[request_id]
                
                # Filter by status
                if request.status != ReviewStatus.PENDING:
                    continue
                    
                # Filter by priority
                if priority and request.priority != priority:
                    continue
                    
                # Check expiration
                if request.is_expired():
                    request.status = ReviewStatus.EXPIRED
                    self.metrics.pending_requests -= 1
                    continue
                    
                pending.append(request)
                
        # Sort by priority and creation time
        priority_order = {
            ReviewPriority.CRITICAL: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3
        }
        
        pending.sort(key=lambda r: (priority_order[r.priority], r.created_at))
        
        return pending
        
    async def submit_review(self,
                           request_id: str,
                           reviewer_id: str,
                           action: ReviewAction,
                           modified_content: Optional[str] = None,
                           notes: Optional[str] = None) -> bool:
        """Submit a review decision."""
        if request_id not in self.reviews:
            return False
            
        request = self.reviews[request_id]
        
        if request.status != ReviewStatus.PENDING:
            return False
            
        # Update request
        request.reviewer_id = reviewer_id
        request.reviewed_at = datetime.now()
        request.action = action
        request.modified_content = modified_content
        request.reviewer_notes = notes
        
        # Update status based on action
        if action == ReviewAction.APPROVE:
            request.status = ReviewStatus.APPROVED
            self.metrics.approved_count += 1
        elif action == ReviewAction.REJECT:
            request.status = ReviewStatus.REJECTED
            self.metrics.rejected_count += 1
        elif action == ReviewAction.MODIFY:
            request.status = ReviewStatus.MODIFIED
            self.metrics.modified_count += 1
        elif action == ReviewAction.ESCALATE:
            request.status = ReviewStatus.ESCALATED
            self.metrics.escalated_count += 1
            
        # Update metrics
        self.metrics.pending_requests -= 1
        
        reviewer_key = reviewer_id
        self.metrics.by_reviewer[reviewer_key] = self.metrics.by_reviewer.get(reviewer_key, 0) + 1
        
        # Calculate review time
        if request.reviewed_at and request.created_at:
            review_time = (request.reviewed_at - request.created_at).total_seconds()
            
            # Update average (simple moving average)
            n = self.metrics.approved_count + self.metrics.rejected_count + self.metrics.modified_count
            self.metrics.average_review_time = (
                (self.metrics.average_review_time * (n - 1) + review_time) / n
            )
            
        logger.info(
            f"Review {request_id} completed by {reviewer_id} with action {action.value}"
        )
        
        return True
        
    def get_metrics(self) -> ReviewMetrics:
        """Get review system metrics."""
        return self.metrics


class HumanInTheLoopSystem:
    """Main human-in-the-loop system for alignment."""
    
    def __init__(self,
                 review_interface: Optional[ReviewInterface] = None,
                 auto_approve_threshold: float = 0.95,
                 review_sample_rate: float = 0.1):
        """
        Initialize HITL system.
        
        Args:
            review_interface: Interface for human reviews
            auto_approve_threshold: Confidence threshold for auto-approval
            review_sample_rate: Rate at which to sample for review
        """
        self.review_interface = review_interface or InMemoryReviewInterface()
        self.auto_approve_threshold = auto_approve_threshold
        self.review_sample_rate = review_sample_rate
        
        # Callbacks for different events
        self.on_review_complete: Optional[Callable] = None
        self.on_content_modified: Optional[Callable] = None
        
    async def should_review(self,
                           prompt: str,
                           response: str,
                           safety_score: Optional[float] = None,
                           context: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if content should be reviewed."""
        # Always review if safety score is low
        if safety_score is not None and safety_score < self.auto_approve_threshold:
            return True
            
        # Sample for review based on rate
        import random
        if random.random() < self.review_sample_rate:
            return True
            
        # Check for specific patterns that require review
        review_patterns = [
            "legal advice",
            "medical advice",
            "financial advice",
            "minors",
            "violence",
            "controversial"
        ]
        
        combined_text = f"{prompt} {response}".lower()
        for pattern in review_patterns:
            if pattern in combined_text:
                return True
                
        return False
        
    async def submit_for_review(self,
                               prompt: str,
                               response: str,
                               priority: ReviewPriority = ReviewPriority.MEDIUM,
                               reason: str = "Content flagged for review",
                               context: Optional[Dict[str, Any]] = None) -> str:
        """Submit content for human review."""
        request = ReviewRequest(
            request_id=str(uuid.uuid4()),
            content_type="both",
            prompt=prompt,
            response=response,
            priority=priority,
            reason=reason,
            context=context or {}
        )
        
        return await self.review_interface.submit_for_review(request)
        
    async def wait_for_review(self,
                             request_id: str,
                             timeout: Optional[float] = None) -> Optional[ReviewRequest]:
        """Wait for review to complete."""
        start_time = datetime.now()
        
        while True:
            status = await self.review_interface.get_review_status(request_id)
            
            if status != ReviewStatus.PENDING:
                result = await self.review_interface.get_review_result(request_id)
                
                # Trigger callback if review is complete
                if result and self.on_review_complete:
                    self.on_review_complete(result)
                    
                return result
                
            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    return None
                    
            # Wait before checking again
            await asyncio.sleep(0.5)
            
    async def get_reviewed_content(self,
                                  request_id: str,
                                  original_response: str) -> str:
        """Get content after review, with modifications if any."""
        result = await self.review_interface.get_review_result(request_id)
        
        if not result:
            return original_response
            
        if result.status == ReviewStatus.APPROVED:
            return original_response
        elif result.status == ReviewStatus.MODIFIED and result.modified_content:
            if self.on_content_modified:
                self.on_content_modified(original_response, result.modified_content)
            return result.modified_content
        elif result.status == ReviewStatus.REJECTED:
            return "This content has been reviewed and cannot be provided."
        else:
            return original_response
            
    def create_review_queue_handler(self) -> 'ReviewQueueHandler':
        """Create a handler for processing review queues."""
        return ReviewQueueHandler(self.review_interface)


class ReviewQueueHandler:
    """Handler for processing review queues."""
    
    def __init__(self, review_interface: ReviewInterface):
        self.review_interface = review_interface
        self.active = False
        
    async def process_reviews(self,
                             reviewer_id: str,
                             review_function: Callable[[ReviewRequest], Tuple[ReviewAction, Optional[str], Optional[str]]]):
        """
        Process pending reviews.
        
        Args:
            reviewer_id: ID of the reviewer
            review_function: Function that takes a ReviewRequest and returns (action, modified_content, notes)
        """
        self.active = True
        
        while self.active:
            # Get pending reviews
            pending = await self.review_interface.get_pending_reviews()
            
            if not pending:
                await asyncio.sleep(1)
                continue
                
            # Process each review
            for request in pending:
                if not self.active:
                    break
                    
                try:
                    # Call review function
                    action, modified_content, notes = review_function(request)
                    
                    # Submit review
                    if isinstance(self.review_interface, InMemoryReviewInterface):
                        await self.review_interface.submit_review(
                            request.request_id,
                            reviewer_id,
                            action,
                            modified_content,
                            notes
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing review {request.request_id}: {e}")
                    
    def stop(self):
        """Stop processing reviews."""
        self.active = False


# Example review functions
def auto_review_safe_content(request: ReviewRequest) -> Tuple[ReviewAction, Optional[str], Optional[str]]:
    """Auto-approve obviously safe content."""
    safe_keywords = ["hello", "thank you", "please help", "how to learn"]
    
    combined_text = f"{request.prompt} {request.response or ''}".lower()
    
    if any(keyword in combined_text for keyword in safe_keywords):
        return ReviewAction.APPROVE, None, "Auto-approved: safe content"
        
    return ReviewAction.ESCALATE, None, "Requires manual review"


def strict_review_policy(request: ReviewRequest) -> Tuple[ReviewAction, Optional[str], Optional[str]]:
    """Strict review policy for sensitive contexts."""
    blocked_terms = ["medical diagnosis", "legal advice", "investment tips"]
    
    combined_text = f"{request.prompt} {request.response or ''}".lower()
    
    for term in blocked_terms:
        if term in combined_text:
            return ReviewAction.REJECT, None, f"Blocked: contains {term}"
            
    return ReviewAction.APPROVE, None, "Passed strict review"