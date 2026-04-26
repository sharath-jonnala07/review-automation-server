"""Main API router aggregating all v1 endpoints."""

from fastapi import APIRouter

from app.api.v1 import products, runs, system

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(products.router, prefix="/products", tags=["products"])
api_router.include_router(runs.router, prefix="/runs", tags=["runs"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
