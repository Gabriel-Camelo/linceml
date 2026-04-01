"""
Módulo de Contratos e Schemas da LinceML.
Define as estruturas de dados de saída padronizadas utilizando Pydantic.
Garante a interoperabilidade e paridade de schema entre os motores Polars e Spark.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class StatisticalSketch(BaseModel):
    """
    Resumo estatístico leve de uma feature.
    Substitui a retenção de dados brutos para evitar Metadata Bloat e mitigar o custo de I/O na rede.
    """
    model_config = ConfigDict(frozen=True)

    min_value: Optional[float] = Field(default=None, description="Valor mínimo encontrado na feature.")
    max_value: Optional[float] = Field(default=None, description="Valor máximo encontrado na feature.")
    mean: Optional[float] = Field(default=None, description="Média aritmética da feature.")
    null_rate: float = Field(default=0.0, description="Percentual de valores nulos (0.0 a 1.0).")
    histogram_bins: Optional[List[float]] = Field(default=None, description="Limites dos decis do histograma.")
    histogram_counts: Optional[List[int]] = Field(default=None, description="Contagem de registros por decil ou bucket.")


class DriftMetricResult(BaseModel):
    """
    Contrato base para o resultado de um teste estatístico específico calculado pelas engines.
    """
    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(..., description="Nome do teste estatístico (ex: 'PSI', 'KS-Test', 'JSD').")
    value: float = Field(..., description="Valor final calculado da métrica.")
    threshold: Optional[float] = Field(default=None, description="Limite aceitável (teto) definido para a métrica.")
    is_drifting: bool = Field(..., description="Flag indicando se a divergência ultrapassou o threshold (alerta).")


class FeatureReport(BaseModel):
    """
    Consolida todas as métricas e os sketches estatísticos comparativos de uma única feature (coluna).
    """
    model_config = ConfigDict(frozen=True)

    feature_name: str = Field(..., description="Nome da coluna analisada.")
    feature_type: str = Field(..., description="Tipo primitivo da coluna (ex: 'numeric', 'categorical').")
    sketch_reference: StatisticalSketch = Field(..., description="Sketch gerado a partir da base de treinamento (referência).")
    sketch_analysis: StatisticalSketch = Field(..., description="Sketch gerado a partir da base de produção (análise).")
    drift_metrics: List[DriftMetricResult] = Field(default_factory=list, description="Lista de testes de drift calculados para esta feature.")


class DriftReport(BaseModel):
    """
    Contrato principal e unificado de saída da LinceML.
    Ponto central do Factory Pattern: independentemente do motor de execução, a saída será sempre este objeto.
    """
    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., description="Identificador lógico do modelo ou dataset monitorado.")
    engine_used: str = Field(..., description="Motor que processou os cálculos ('polars' ou 'spark').")
    execution_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data e hora exata da geração do relatório (UTC)."
    )
    total_records_reference: int = Field(..., description="Volume total de linhas da base de referência.")
    total_records_analysis: int = Field(..., description="Volume total de linhas da amostra analisada.")
    
    features: Dict[str, FeatureReport] = Field(..., description="Dicionário com os relatórios detalhados mapeados pelo nome da feature.")

    def to_dict(self) -> dict:
        """
        Exporta o relatório para um dicionário Python nativo estruturado.
        Ideal para uso imediato em memória durante a execução de pipelines MLOps.
        """
        return self.model_dump(mode='python')

    def to_json(self) -> str:
        """
        Exporta o relatório consolidado para uma string JSON formatada.
        Utiliza o backend em Rust do Pydantic para alta performance.
        Ideal para persistência em Data Lakes ou ingestão por ferramentas de observabilidade.
        """
        return self.model_dump_json(indent=2)