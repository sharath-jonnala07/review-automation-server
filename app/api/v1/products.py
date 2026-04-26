"""Products API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.types import ProductKey
from app.db.models import Product as ProductORM

router = APIRouter()


class ProductPayload(BaseModel):
    """Product create/update request."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    key: str = Field(..., min_length=2, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(..., min_length=1, max_length=160)
    appstore_id: str | None = None
    play_package: str | None = None
    gdoc_id: str | None = None
    gmail_to: str | None = None
    is_active: bool = True

    @model_validator(mode="after")
    def validate_store_identifier(self) -> "ProductPayload":
        if not self.appstore_id and not self.play_package:
            raise ValueError("At least one store identifier is required")
        return self


class ProductUpdatePayload(BaseModel):
    """Partial product update request."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    display_name: str | None = Field(default=None, min_length=1, max_length=160)
    appstore_id: str | None = None
    play_package: str | None = None
    gdoc_id: str | None = None
    gmail_to: str | None = None
    is_active: bool | None = None


class ProductResponse(BaseModel):
    """Product API response."""

    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    key: ProductKey
    display_name: str
    appstore_id: str | None = None
    play_package: str | None = None
    gdoc_id: str | None = None
    gmail_to: str | None = None
    is_active: bool = True
    created_at: datetime


def _to_response(row: ProductORM) -> ProductResponse:
    return ProductResponse(
        key=ProductKey(row.key),
        display_name=row.display_name,
        appstore_id=row.appstore_id,
        play_package=row.play_package,
        gdoc_id=row.gdoc_id,
        gmail_to=row.gmail_to,
        is_active=row.is_active,
        created_at=row.created_at,
    )


@router.get("", response_model=list[ProductResponse])
async def list_products(
    db: AsyncSession = Depends(get_db),
) -> list[ProductResponse]:
    """List all tracked products."""
    result = await db.execute(select(ProductORM).order_by(ProductORM.display_name.asc()))
    rows = result.scalars().all()
    return [_to_response(row) for row in rows]


@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    payload: ProductPayload,
    db: AsyncSession = Depends(get_db),
) -> ProductResponse:
    """Create a tracked product."""
    existing = await db.get(ProductORM, payload.key)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Product key already exists")

    row = ProductORM(
        key=payload.key,
        display_name=payload.display_name,
        appstore_id=payload.appstore_id,
        play_package=payload.play_package,
        gdoc_id=payload.gdoc_id,
        gmail_to=payload.gmail_to,
        is_active=payload.is_active,
    )
    db.add(row)
    await db.flush()
    return _to_response(row)


@router.get("/{product_key}", response_model=ProductResponse)
async def get_product(
    product_key: str,
    db: AsyncSession = Depends(get_db),
) -> ProductResponse:
    """Get a single product by key."""
    row = await db.get(ProductORM, product_key)
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return _to_response(row)


@router.put("/{product_key}", response_model=ProductResponse)
async def replace_product(
    product_key: str,
    payload: ProductPayload,
    db: AsyncSession = Depends(get_db),
) -> ProductResponse:
    """Replace a tracked product."""
    if payload.key != product_key:
        raise HTTPException(status_code=400, detail="Product key cannot be changed")
    row = await db.get(ProductORM, product_key)
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")

    row.display_name = payload.display_name
    row.appstore_id = payload.appstore_id
    row.play_package = payload.play_package
    row.gdoc_id = payload.gdoc_id
    row.gmail_to = payload.gmail_to
    row.is_active = payload.is_active
    await db.flush()
    return _to_response(row)


@router.patch("/{product_key}", response_model=ProductResponse)
async def update_product(
    product_key: str,
    payload: ProductUpdatePayload,
    db: AsyncSession = Depends(get_db),
) -> ProductResponse:
    """Partially update a tracked product."""
    row = await db.get(ProductORM, product_key)
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")

    update = payload.model_dump(exclude_unset=True)
    next_appstore_id = update.get("appstore_id", row.appstore_id)
    next_play_package = update.get("play_package", row.play_package)
    if not next_appstore_id and not next_play_package:
        raise HTTPException(status_code=422, detail="At least one store identifier is required")

    for key, value in update.items():
        setattr(row, key, value)
    await db.flush()
    return _to_response(row)


@router.delete("/{product_key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(
    product_key: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Delete a tracked product and its dependent local data."""
    row = await db.get(ProductORM, product_key)
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")
    await db.delete(row)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
