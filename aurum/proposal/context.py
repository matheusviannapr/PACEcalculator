"""
Contexto da proposta: o pacote de dados que alimenta o documento.

Reúne, para um telhado, tudo que a proposta precisa -- lead e empresa,
recurso solar, layout, sistema elétrico, economia -- num objeto único e
serializável. Assim o gerador de LaTeX não precisa conhecer nenhum dos
módulos de cálculo, e o mesmo contexto serve para o JSON, o CSV e a interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..geo.roofs import RoofLead
from ..pv.consumption import EstimativaConsumo
from ..pv.financials import ResultadoEconomico
from ..pv.layout import LayoutModulos
from ..pv.sizing import SistemaFV
from ..pv.solar import PerfilGeracao


@dataclass
class DadosEmissor:
    """Quem assina a proposta. Preenchido pela configuração da empresa."""

    empresa: str = "PACE Inteligência Energética"
    cnpj: str | None = None
    responsavel_tecnico: str | None = None
    crea: str | None = None
    telefone: str | None = None
    email: str | None = None
    site: str | None = None
    cidade: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ContextoProposta:
    """Tudo que uma proposta precisa saber sobre um telhado."""

    lead: RoofLead
    perfil_solar: PerfilGeracao
    layout: LayoutModulos
    sistema: SistemaFV
    consumo: EstimativaConsumo
    economia: ResultadoEconomico
    emissor: DadosEmissor = field(default_factory=DadosEmissor)
    data_emissao: date = field(default_factory=date.today)
    validade_dias: int = 30
    codigo: str | None = None
    #: Origem dos dados de consumo: "estimado" ou "fatura".
    origem_consumo: str = "estimado"
    observacoes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def titulo_cliente(self) -> str:
        """Nome comercial do prospect, com a melhor fonte disponível."""
        return self.lead.company.get("nome") or self.lead.name

    @property
    def endereco(self) -> str:
        return self.lead.company.get("endereco") or "endereço não identificado"

    @property
    def referencia(self) -> str:
        """Código da proposta, derivado do id OSM quando não informado."""
        if self.codigo:
            return self.codigo
        return f"AUR-{self.lead.osm_type[:1].upper()}{self.lead.osm_id}"

    @property
    def cobertura_consumo(self) -> float | None:
        """Fração do consumo estimado que a geração cobre."""
        if not self.consumo.consumo_anual_kwh:
            return None
        geracao = self.sistema.geracao_anual_kwh or 0.0
        return geracao / self.consumo.consumo_anual_kwh

    def avisos_consolidados(self) -> list[str]:
        """
        Todos os avisos das etapas, sem repetição e em ordem de relevância.

        A proposta precisa carregar isso de forma visível: o dado de origem é
        um mapa colaborativo, não um levantamento de campo.
        """
        avisos: list[str] = []

        if self.origem_consumo == "estimado":
            avisos.append(
                "O consumo de energia foi estimado a partir do tipo e do porte da edificação, "
                "não de fatura. Confirmar com a conta de luz antes de fechar valores."
            )
        if not self.perfil_solar.confiavel and self.perfil_solar.aviso:
            avisos.append(self.perfil_solar.aviso)
        if self.lead.company.get("fonte", "").startswith("osm_poi_proximo"):
            avisos.append(
                "Os dados da empresa vieram de um ponto de interesse próximo ao telhado, "
                "não de registro na própria edificação. Confirmar o ocupante antes do contato."
            )

        avisos.extend(self.layout.avisos)
        avisos.extend(self.sistema.avisos)
        avisos.extend(self.consumo.avisos)
        avisos.extend(self.economia.avisos)
        avisos.extend(self.observacoes)

        avisos.append(
            "Geometria de telhado obtida do OpenStreetMap. A área é a projeção horizontal da "
            "edificação e não considera obstruções, sombreamento do entorno nem condições "
            "estruturais da cobertura."
        )
        # dict.fromkeys preserva a ordem e elimina repetição
        return list(dict.fromkeys(a for a in avisos if a))

    def resumo(self) -> dict[str, Any]:
        """Linha achatada para índice em lote, CSV e planilha."""
        return {
            "referencia": self.referencia,
            "lead_id": self.lead.lead_id,
            "cliente": self.titulo_cliente,
            "tipo_edificacao": self.lead.building_type,
            "endereco": self.endereco,
            "telefone": self.lead.company.get("telefone"),
            "site": self.lead.company.get("site"),
            "email": self.lead.company.get("email"),
            "latitude": round(self.lead.centroid_lat, 6),
            "longitude": round(self.lead.centroid_lon, 6),
            "area_telhado_m2": round(self.lead.metrics.area_m2, 0),
            "score_lead": self.lead.score,
            "modulos": self.sistema.modulos_totais,
            "potencia_kwp": round(self.sistema.potencia_cc_kwp, 1),
            "geracao_anual_kwh": round(self.sistema.geracao_anual_kwh or 0, 0),
            "consumo_anual_estimado_kwh": round(self.consumo.consumo_anual_kwh, 0),
            "cobertura_consumo": (
                round(self.cobertura_consumo, 3) if self.cobertura_consumo is not None else None
            ),
            "capex_brl": round(self.economia.premissas.capex_brl, 2),
            "economia_ano1_brl": round(self.economia.economia_ano1_brl, 2),
            "payback_anos": (
                round(self.economia.payback_simples_anos, 2)
                if self.economia.payback_simples_anos
                else None
            ),
            "tir_percent": (
                round(self.economia.tir_anual * 100, 1) if self.economia.tir_anual is not None else None
            ),
            "vpl_brl": round(self.economia.vpl_brl, 2),
            "fonte_irradiacao": self.perfil_solar.fonte,
            "osm_url": self.lead.osm_url,
            "maps_url": self.lead.maps_url,
        }

    def as_dict(self) -> dict[str, Any]:
        """Serialização completa, para o JSON que acompanha a proposta."""
        return {
            "referencia": self.referencia,
            "data_emissao": self.data_emissao.isoformat(),
            "validade_dias": self.validade_dias,
            "emissor": self.emissor.as_dict(),
            "lead": self.lead.to_record(),
            "tags_osm": self.lead.tags,
            "empresa": self.lead.company,
            "recurso_solar": self.perfil_solar.as_dict(),
            "layout": self.layout.as_dict(),
            "sistema": self.sistema.as_dict(),
            "consumo": self.consumo.as_dict(),
            "economia": self.economia.as_dict(),
            "fluxo_caixa": [
                {
                    "ano": l.ano,
                    "ano_calendario": l.ano_calendario,
                    "geracao_kwh": round(l.geracao_kwh, 0),
                    "tarifa_brl_kwh": round(l.tarifa_brl_kwh, 4),
                    "economia_autoconsumo_brl": round(l.economia_autoconsumo_brl, 2),
                    "economia_injecao_brl": round(l.economia_injecao_brl, 2),
                    "custo_fio_b_brl": round(l.custo_fio_b_brl, 2),
                    "opex_brl": round(l.opex_brl, 2),
                    "fluxo_liquido_brl": round(l.fluxo_liquido_brl, 2),
                    "vpl_acumulado_brl": round(l.vpl_acumulado_brl, 2),
                }
                for l in self.economia.fluxo
            ],
            "avisos": self.avisos_consolidados(),
        }
