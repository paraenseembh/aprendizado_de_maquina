# Projeto 2 — Hipertensão e Hiperparâmetros (PNS 2019)

Trabalho prático da disciplina de **Aprendizado de Máquina** — Ciência de Dados e
Inteligência Artificial, PUC Minas.

**Autor:** Lucas do Nascimento

📄 **Artigo:** [`artigo/hipertensao_e_hiperparametros_pns2019.pdf`](artigo/hipertensao_e_hiperparametros_pns2019.pdf)
(*Hipertensão e Hiperparâmetros: aplicação de aprendizado de máquina na Pesquisa Nacional de Saúde 2019*, 3 p., formato ACM, nov./2025, versão preliminar)

🗂️ **Base de dados:** Pesquisa Nacional de Saúde (PNS) 2019, IBGE. Veja [`REFERENCIA_BASE.md`](REFERENCIA_BASE.md)

---

## Objetivo

Prever se uma pessoa tem **hipertensão arterial** (variável `Q00201`: *"Algum médico
já lhe deu o diagnóstico de hipertensão arterial?"*) a partir de atributos da PNS 2019,
e comparar algoritmos de classificação e ajustes de hiperparâmetros.

## Metodologia (resumo do artigo)

1. **Entendimento do domínio com o método CAPTO** (Zárate et al., 2023). O problema foi dividido em quatro dimensões:
   características do indivíduo · saúde física e mental · genética · qualidade de vida.
2. **Seleção de atributos** da PNS a partir desse mapa conceitual.
3. **Pré-processamento** (KNIME):
   - *Fix Value*: valores ausentes de "trabalhava de madrugada?", "trabalhava 24h seguidas?" e "já fumou no passado?" preenchidos com "Não";
   - *Remove Row*: linhas sem `classificacao_imc` foram removidas;
   - combinação de atributos e discretização de faixas;
   - **undersampling** no treino (13.451 × 13.451). O teste usa a base desbalanceada.
4. **Modelagem:** Árvore de Decisão e Random Forest, com ajuste de hiperparâmetros
   (Grid Search / AutoML). O workflow também tem nós de AdaBoost, XGBoost, H2O AutoML,
   SMOTE e K-Means (k=7) + t-SNE.

## Resultados preliminares (do artigo)

| Modelo | Classe | Recall | Precision | F-measure |
|---|---|---|---|---|
| Árvore de Decisão | Não | 0,625 | 0,714 | 0,667 |
| Árvore de Decisão | Sim | 0,577 | 0,476 | 0,522 |
| Random Forest | Não | 0,624 | 0,713 | 0,665 |
| Random Forest | Sim | 0,575 | 0,475 | 0,520 |

As duas abordagens têm desempenho praticamente igual. Com os atributos disponíveis,
o poder preditivo é modesto (F1 ≈ 0,52 na classe hipertenso). As seções de Discussão
e Conclusão do artigo ainda estão em andamento.

## Onde está cada parte no repositório

| Item | Caminho |
|---|---|
| Artigo (PDF) | [`artigo/`](artigo/) |
| Referência e dicionário da base | [`REFERENCIA_BASE.md`](REFERENCIA_BASE.md) |
| Workflow KNIME (pasta do workspace) | [`../mineracao_de_dados/`](../mineracao_de_dados/) |
| Workflow KNIME exportado | [`../KNIME_DECISIONTREEproject1.knwf`](../KNIME_DECISIONTREEproject1.knwf) |
| Base PNS pré-processada (CSV `;`) | [`../c45/dados_c45_sembalanceamento_originais.csv`](../c45/dados_c45_sembalanceamento_originais.csv) |
| Implementação própria do C4.5 | [`../c45/`](../c45/) |
| Registro de uso de IA | [`REGISTRO_USO_IA.md`](REGISTRO_USO_IA.md) |

> Os arquivos do KNIME e da base ficaram nos caminhos originais porque o workflow
> lê o CSV por caminho absoluto. Movê-los quebraria o `CSV Reader`.

## Como reproduzir

1. Baixe os microdados da PNS 2019 (links em [`REFERENCIA_BASE.md`](REFERENCIA_BASE.md))
   ou use direto o CSV já filtrado em `c45/`.
2. No KNIME Analytics Platform: *File → Import KNIME Workflow…* e escolha
   `KNIME_DECISIONTREEproject1.knwf` (ou abra a pasta `mineracao_de_dados/` como workflow).
3. No nó **CSV Reader (#1)**, aponte o caminho para
   `c45/dados_c45_sembalanceamento_originais.csv` na sua máquina (separador `;`).
4. Execute o workflow (*Execute all*).

## Referência do trabalho

```
NASCIMENTO, Lucas do. Hipertensão e Hiperparâmetros: aplicação de aprendizado de
máquina na Pesquisa Nacional de Saúde 2019. Trabalho prático (Aprendizado de Máquina),
Pontifícia Universidade Católica de Minas Gerais, Belo Horizonte, 2025.
```
