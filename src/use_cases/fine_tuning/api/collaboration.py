"""
Collaboration features for team-based experiment management
Includes comments, sharing, team workspaces, and notifications
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborationType(Enum):
    COMMENT = "comment"
    MENTION = "mention"
    SHARE = "share"
    REVIEW = "review"
    APPROVAL = "approval"


class NotificationType(Enum):
    COMMENT = "comment"
    MENTION = "mention"
    EXPERIMENT_SHARED = "experiment_shared"
    EXPERIMENT_COMPLETED = "experiment_completed"
    DEPLOYMENT_READY = "deployment_ready"
    REVIEW_REQUESTED = "review_requested"
    APPROVAL_NEEDED = "approval_needed"
    TRAINING_FAILED = "training_failed"


class WorkspaceRole(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


@dataclass
class Comment:
    """Represents a comment on an experiment or component"""

    id: str
    experiment_id: str
    user_id: str
    username: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime]
    parent_id: Optional[str]  # For threaded comments
    mentions: List[str]
    resolved: bool
    reactions: Dict[str, List[str]]  # emoji -> list of user_ids
    attachments: List[str]


@dataclass
class TeamWorkspace:
    """Represents a team workspace"""

    id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    members: Dict[str, WorkspaceRole]  # user_id -> role
    experiments: List[str]
    datasets: List[str]
    settings: Dict[str, Any]
    is_public: bool


@dataclass
class ShareLink:
    """Represents a share link for an experiment"""

    id: str
    experiment_id: str
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime]
    access_level: str  # "view", "comment", "edit"
    password: Optional[str]
    max_uses: Optional[int]
    used_count: int
    is_active: bool


@dataclass
class ReviewRequest:
    """Represents a review request for an experiment"""

    id: str
    experiment_id: str
    requester_id: str
    reviewers: List[str]
    title: str
    description: str
    created_at: datetime
    due_date: Optional[datetime]
    status: str  # "pending", "in_review", "approved", "rejected"
    reviews: List[Dict[str, Any]]
    approval_required: bool


@dataclass
class Notification:
    """Represents a user notification"""

    id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    data: Dict[str, Any]
    created_at: datetime
    read: bool
    action_url: Optional[str]


class CollaborationManager:
    """Manages collaboration features"""

    def __init__(self):
        self.comments_db: Dict[str, List[Comment]] = {}
        self.workspaces_db: Dict[str, TeamWorkspace] = {}
        self.share_links_db: Dict[str, ShareLink] = {}
        self.review_requests_db: Dict[str, ReviewRequest] = {}
        self.notifications_db: Dict[str, List[Notification]] = {}
        self.user_workspaces: Dict[str, Set[str]] = {}  # user_id -> workspace_ids

    # =====================
    # Comment Management
    # =====================

    def add_comment(
        self,
        experiment_id: str,
        user_id: str,
        username: str,
        content: str,
        parent_id: Optional[str] = None,
        mentions: List[str] = None,
        attachments: List[str] = None,
    ) -> Comment:
        """Add a comment to an experiment"""

        comment = Comment(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            user_id=user_id,
            username=username,
            content=content,
            created_at=datetime.utcnow(),
            updated_at=None,
            parent_id=parent_id,
            mentions=mentions or [],
            resolved=False,
            reactions={},
            attachments=attachments or [],
        )

        # Store comment
        if experiment_id not in self.comments_db:
            self.comments_db[experiment_id] = []
        self.comments_db[experiment_id].append(comment)

        # Send notifications for mentions
        for mentioned_user in comment.mentions:
            self._send_notification(
                user_id=mentioned_user,
                notification_type=NotificationType.MENTION,
                title=f"{username} mentioned you",
                message=f"In experiment {experiment_id}: {content[:100]}...",
                data={"comment_id": comment.id, "experiment_id": experiment_id},
            )

        logger.info(f"Comment added to experiment {experiment_id} by {username}")

        return comment

    def get_comments(self, experiment_id: str, include_resolved: bool = True) -> List[Comment]:
        """Get comments for an experiment"""

        comments = self.comments_db.get(experiment_id, [])

        if not include_resolved:
            comments = [c for c in comments if not c.resolved]

        # Sort by creation time
        comments.sort(key=lambda c: c.created_at)

        return comments

    def resolve_comment(self, comment_id: str, experiment_id: str) -> bool:
        """Mark a comment as resolved"""

        if experiment_id in self.comments_db:
            for comment in self.comments_db[experiment_id]:
                if comment.id == comment_id:
                    comment.resolved = True
                    comment.updated_at = datetime.utcnow()
                    return True

        return False

    def add_reaction(self, comment_id: str, experiment_id: str, user_id: str, emoji: str) -> bool:
        """Add a reaction to a comment"""

        if experiment_id in self.comments_db:
            for comment in self.comments_db[experiment_id]:
                if comment.id == comment_id:
                    if emoji not in comment.reactions:
                        comment.reactions[emoji] = []

                    if user_id not in comment.reactions[emoji]:
                        comment.reactions[emoji].append(user_id)

                    return True

        return False

    # =====================
    # Workspace Management
    # =====================

    def create_workspace(
        self, name: str, description: str, created_by: str, is_public: bool = False
    ) -> TeamWorkspace:
        """Create a new team workspace"""

        workspace = TeamWorkspace(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_by=created_by,
            created_at=datetime.utcnow(),
            members={created_by: WorkspaceRole.OWNER},
            experiments=[],
            datasets=[],
            settings={},
            is_public=is_public,
        )

        self.workspaces_db[workspace.id] = workspace

        # Track user's workspaces
        if created_by not in self.user_workspaces:
            self.user_workspaces[created_by] = set()
        self.user_workspaces[created_by].add(workspace.id)

        logger.info(f"Workspace '{name}' created by {created_by}")

        return workspace

    def add_workspace_member(
        self, workspace_id: str, user_id: str, role: WorkspaceRole = WorkspaceRole.MEMBER
    ) -> bool:
        """Add a member to a workspace"""

        if workspace_id in self.workspaces_db:
            workspace = self.workspaces_db[workspace_id]
            workspace.members[user_id] = role

            # Track user's workspaces
            if user_id not in self.user_workspaces:
                self.user_workspaces[user_id] = set()
            self.user_workspaces[user_id].add(workspace_id)

            # Send notification
            self._send_notification(
                user_id=user_id,
                notification_type=NotificationType.EXPERIMENT_SHARED,
                title="Added to workspace",
                message=f"You've been added to workspace '{workspace.name}'",
                data={"workspace_id": workspace_id},
            )

            return True

        return False

    def get_user_workspaces(self, user_id: str) -> List[TeamWorkspace]:
        """Get all workspaces a user belongs to"""

        workspace_ids = self.user_workspaces.get(user_id, set())
        workspaces = []

        for ws_id in workspace_ids:
            if ws_id in self.workspaces_db:
                workspaces.append(self.workspaces_db[ws_id])

        return workspaces

    def add_experiment_to_workspace(self, workspace_id: str, experiment_id: str) -> bool:
        """Add an experiment to a workspace"""

        if workspace_id in self.workspaces_db:
            workspace = self.workspaces_db[workspace_id]

            if experiment_id not in workspace.experiments:
                workspace.experiments.append(experiment_id)

            # Notify workspace members
            for member_id in workspace.members:
                self._send_notification(
                    user_id=member_id,
                    notification_type=NotificationType.EXPERIMENT_SHARED,
                    title="New experiment in workspace",
                    message=f"Experiment {experiment_id} added to '{workspace.name}'",
                    data={"workspace_id": workspace_id, "experiment_id": experiment_id},
                )

            return True

        return False

    # =====================
    # Sharing Management
    # =====================

    def create_share_link(
        self,
        experiment_id: str,
        created_by: str,
        access_level: str = "view",
        expires_in_hours: Optional[int] = None,
        password: Optional[str] = None,
        max_uses: Optional[int] = None,
    ) -> ShareLink:
        """Create a shareable link for an experiment"""

        expires_at = None
        if expires_in_hours:
            from datetime import timedelta

            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)

        share_link = ShareLink(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            created_by=created_by,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            access_level=access_level,
            password=password,
            max_uses=max_uses,
            used_count=0,
            is_active=True,
        )

        self.share_links_db[share_link.id] = share_link

        logger.info(f"Share link created for experiment {experiment_id}")

        return share_link

    def use_share_link(
        self, link_id: str, password: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Use a share link to access an experiment"""

        if link_id not in self.share_links_db:
            return None

        link = self.share_links_db[link_id]

        # Check if link is active
        if not link.is_active:
            return None

        # Check expiration
        if link.expires_at and datetime.utcnow() > link.expires_at:
            link.is_active = False
            return None

        # Check password
        if link.password and link.password != password:
            return None

        # Check usage limit
        if link.max_uses and link.used_count >= link.max_uses:
            link.is_active = False
            return None

        # Increment usage
        link.used_count += 1

        return {"experiment_id": link.experiment_id, "access_level": link.access_level}

    # =====================
    # Review Management
    # =====================

    def create_review_request(
        self,
        experiment_id: str,
        requester_id: str,
        reviewers: List[str],
        title: str,
        description: str,
        due_date: Optional[datetime] = None,
        approval_required: bool = False,
    ) -> ReviewRequest:
        """Create a review request for an experiment"""

        review_request = ReviewRequest(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            requester_id=requester_id,
            reviewers=reviewers,
            title=title,
            description=description,
            created_at=datetime.utcnow(),
            due_date=due_date,
            status="pending",
            reviews=[],
            approval_required=approval_required,
        )

        self.review_requests_db[review_request.id] = review_request

        # Notify reviewers
        for reviewer_id in reviewers:
            self._send_notification(
                user_id=reviewer_id,
                notification_type=NotificationType.REVIEW_REQUESTED,
                title="Review requested",
                message=f"{title}: {description[:100]}...",
                data={"review_request_id": review_request.id, "experiment_id": experiment_id},
            )

        logger.info(f"Review request created for experiment {experiment_id}")

        return review_request

    def submit_review(
        self,
        review_request_id: str,
        reviewer_id: str,
        approved: bool,
        comments: str,
        suggestions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Submit a review for a review request"""

        if review_request_id not in self.review_requests_db:
            return False

        review_request = self.review_requests_db[review_request_id]

        if reviewer_id not in review_request.reviewers:
            return False

        # Add review
        review = {
            "reviewer_id": reviewer_id,
            "approved": approved,
            "comments": comments,
            "suggestions": suggestions or {},
            "submitted_at": datetime.utcnow().isoformat(),
        }

        review_request.reviews.append(review)

        # Update status
        if review_request.approval_required:
            all_approved = all(r["approved"] for r in review_request.reviews)
            if all_approved and len(review_request.reviews) == len(review_request.reviewers):
                review_request.status = "approved"
            elif not approved:
                review_request.status = "rejected"
            else:
                review_request.status = "in_review"
        else:
            review_request.status = "in_review"

        # Notify requester
        self._send_notification(
            user_id=review_request.requester_id,
            notification_type=NotificationType.COMMENT,
            title="Review submitted",
            message=f"Review submitted for '{review_request.title}'",
            data={"review_request_id": review_request_id, "approved": approved},
        )

        return True

    # =====================
    # Notification Management
    # =====================

    def _send_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        title: str,
        message: str,
        data: Dict[str, Any] = None,
        action_url: Optional[str] = None,
    ):
        """Send a notification to a user"""

        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=notification_type,
            title=title,
            message=message,
            data=data or {},
            created_at=datetime.utcnow(),
            read=False,
            action_url=action_url,
        )

        if user_id not in self.notifications_db:
            self.notifications_db[user_id] = []

        self.notifications_db[user_id].append(notification)

        # In production, also send real-time notification via WebSocket
        logger.info(f"Notification sent to user {user_id}: {title}")

    def get_user_notifications(
        self, user_id: str, unread_only: bool = False, limit: int = 50
    ) -> List[Notification]:
        """Get notifications for a user"""

        notifications = self.notifications_db.get(user_id, [])

        if unread_only:
            notifications = [n for n in notifications if not n.read]

        # Sort by creation time (newest first)
        notifications.sort(key=lambda n: n.created_at, reverse=True)

        return notifications[:limit]

    def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """Mark a notification as read"""

        if user_id in self.notifications_db:
            for notification in self.notifications_db[user_id]:
                if notification.id == notification_id:
                    notification.read = True
                    return True

        return False

    def mark_all_notifications_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user"""

        count = 0
        if user_id in self.notifications_db:
            for notification in self.notifications_db[user_id]:
                if not notification.read:
                    notification.read = True
                    count += 1

        return count

    # =====================
    # Activity Feed
    # =====================

    def get_activity_feed(
        self, user_id: str, workspace_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get activity feed for a user or workspace"""

        activities = []

        # Get relevant experiment IDs
        experiment_ids = []

        if workspace_id and workspace_id in self.workspaces_db:
            experiment_ids = self.workspaces_db[workspace_id].experiments
        else:
            # Get experiments from all user's workspaces
            for ws_id in self.user_workspaces.get(user_id, []):
                if ws_id in self.workspaces_db:
                    experiment_ids.extend(self.workspaces_db[ws_id].experiments)

        # Collect activities (comments, reviews, etc.)
        for exp_id in experiment_ids:
            # Add comments as activities
            for comment in self.comments_db.get(exp_id, []):
                activities.append(
                    {
                        "type": "comment",
                        "timestamp": comment.created_at,
                        "user": comment.username,
                        "experiment_id": exp_id,
                        "content": comment.content[:100],
                        "data": asdict(comment),
                    }
                )

        # Add review requests as activities
        for review_request in self.review_requests_db.values():
            if review_request.experiment_id in experiment_ids:
                activities.append(
                    {
                        "type": "review_request",
                        "timestamp": review_request.created_at,
                        "user": review_request.requester_id,
                        "experiment_id": review_request.experiment_id,
                        "content": review_request.title,
                        "data": asdict(review_request),
                    }
                )

        # Sort by timestamp (newest first)
        activities.sort(key=lambda a: a["timestamp"], reverse=True)

        return activities[:limit]


# Example usage
if __name__ == "__main__":
    collab = CollaborationManager()

    # Create workspace
    workspace = collab.create_workspace(
        name="ML Team", description="Machine Learning Team Workspace", created_by="user1"
    )

    # Add members
    collab.add_workspace_member(workspace.id, "user2", WorkspaceRole.MEMBER)
    collab.add_workspace_member(workspace.id, "user3", WorkspaceRole.VIEWER)

    # Add experiment to workspace
    collab.add_experiment_to_workspace(workspace.id, "exp_001")

    # Add comment with mention
    comment = collab.add_comment(
        experiment_id="exp_001",
        user_id="user1",
        username="Alice",
        content="@Bob Please review the hyperparameters",
        mentions=["user2"],
    )

    # Create review request
    review_request = collab.create_review_request(
        experiment_id="exp_001",
        requester_id="user1",
        reviewers=["user2", "user3"],
        title="Review fine-tuning configuration",
        description="Please review the configuration for the customer support model",
        approval_required=True,
    )

    # Submit review
    collab.submit_review(
        review_request_id=review_request.id,
        reviewer_id="user2",
        approved=True,
        comments="Looks good! Minor suggestion on learning rate.",
        suggestions={"learning_rate": 1e-5},
    )

    # Get activity feed
    activities = collab.get_activity_feed("user1")
    print(f"Activity feed: {len(activities)} items")
