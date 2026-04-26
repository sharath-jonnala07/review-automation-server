"""SQLAlchemy ORM models mapping to the database schema."""

from datetime import date, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Product(Base):
    """Tracked products configuration."""

    __tablename__ = "products"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    appstore_id: Mapped[str | None] = mapped_column(String)
    play_package: Mapped[str | None] = mapped_column(String)
    gdoc_id: Mapped[str | None] = mapped_column(String)
    gmail_to: Mapped[str | None] = mapped_column(String)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="product", cascade="all, delete-orphan")
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )


class Review(Base):
    """Raw reviews from App Store or Play Store."""

    __tablename__ = "reviews"
    __table_args__ = (UniqueConstraint("product_key", "source", "external_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    product_key: Mapped[str] = mapped_column(ForeignKey("products.key"), nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    posted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    version: Mapped[str | None] = mapped_column(String)
    language: Mapped[str] = mapped_column(String, default="en")
    country: Mapped[str] = mapped_column(String, default="in")
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False)

    product: Mapped["Product"] = relationship(back_populates="reviews")
    embedding: Mapped["ReviewEmbedding | None"] = relationship(
        back_populates="review", uselist=False, cascade="all, delete-orphan"
    )


class ReviewEmbedding(Base):
    """Vector embeddings for reviews (sqlite-vec)."""

    __tablename__ = "review_embeddings"

    review_id: Mapped[str] = mapped_column(
        ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[bytes] = mapped_column(nullable=False)

    review: Mapped["Review"] = relationship(back_populates="embedding")


class Run(Base):
    """Agent runs with audit trail."""

    __tablename__ = "runs"
    __table_args__ = (UniqueConstraint("product_key", "iso_week"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    product_key: Mapped[str] = mapped_column(ForeignKey("products.key"), nullable=False)
    iso_week: Mapped[str] = mapped_column(String, nullable=False)
    window_start: Mapped[date] = mapped_column(nullable=False)
    window_end: Mapped[date] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    metrics_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    gdoc_id: Mapped[str | None] = mapped_column(String)
    gdoc_heading_id: Mapped[str | None] = mapped_column(String)
    gmail_message_id: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    product: Mapped["Product"] = relationship(back_populates="runs")
    themes: Mapped[list["Theme"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Theme(Base):
    """Generated themes per run."""

    __tablename__ = "themes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str] = mapped_column(String, nullable=False)
    review_count: Mapped[int] = mapped_column(Integer, nullable=False)
    representative_review_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    action_ideas_json: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped["Run"] = relationship(back_populates="themes")


class PromptVersion(Base):
    """Version-controlled prompts."""

    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("name", "version_hash"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    version_hash: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_tested: Mapped[str | None] = mapped_column(String)
    performance_score: Mapped[float | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """Audit trail for every significant operation."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    event_data: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@event.listens_for(Run, "before_update")
def receive_before_update(_mapper: object, _connection: object, target: Run) -> None:
    """Auto-update updated_at on Run modifications."""
    target.updated_at = datetime.utcnow()
