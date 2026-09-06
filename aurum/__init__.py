"""
Aurum — Prospecção e proposta automatizada de geração solar fotovoltaica.

Unifica dois fluxos que antes eram programas separados:

* **Prospecção** (`aurum.geo`): a partir de uma região (nome ou bbox), varre o
  OpenStreetMap via Overpass, identifica os maiores telhados aproveitáveis,
  qualifica-os por geometria e enriquece cada um com os dados da empresa
  ocupante (nome, site, telefone, endereço).
* **Dimensionamento e proposta** (`aurum.pv` / `aurum.proposal`): para cada
  telhado selecionado, calcula o arranjo fotovoltaico que cabe na área real,
  estima geração via PVWatts, monta a análise econômica e gera a proposta
  em LaTeX.

O ponto de entrada de alto nível é `aurum.pipeline`.
"""

__version__ = "3.0.0"
__all__ = ["__version__"]
