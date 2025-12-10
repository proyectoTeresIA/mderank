from pydantic import BaseModel, Field, validator
from typing import Literal

from pydantic import BaseModel, Field
from .logger import create_logger
import logging


class MDERankConfig(BaseModel):
    """
    Configuración para el extractor MDERank.
    Esta configuración NO contiene el modelo en sí, solo parámetros.
    """
    lang: Literal["es", "en"] = Field(
        default="es", description="Idioma del modelo."
    )

    model_name_or_path: str = Field(
        default="PlanTL-GOB-ES/roberta-base-bne",
        description="Modelo HuggingFace a usar para embeddings."
    )

    use_cuda: bool = Field(
        default=False,
        description="Activa CUDA si está disponible."
    )

    model_type: Literal["bert", "roberta"] = Field(
        default="bert", description="Arquitectura del modelo."
    )

    doc_embed_mode: Literal["mean", "max"] = Field(
        default="mean",
        description="Método para obtener embedding de documento."
    )

    batch_size: int = Field(
        default=1,
        description="Tamaño de lote para procesamiento."
    )

    layer_num: int = Field(
        default=-1,
        description="Capa oculta del modelo a usar."
    )

    log_level: str = "INFO"

    logger: logging.Logger = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Crear logger automáticamente
        self.logger = create_logger(level=self.log_level)

