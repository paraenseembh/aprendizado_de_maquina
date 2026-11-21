# Referência da base de dados — PNS 2019

## Fonte

**Pesquisa Nacional de Saúde (PNS) 2019**, realizada pelo Instituto Brasileiro de
Geografia e Estatística (IBGE) em convênio com o Ministério da Saúde.

- Página oficial da pesquisa (IBGE): https://www.ibge.gov.br/estatisticas/sociais/saude/9160-pesquisa-nacional-de-saude.html
- Microdados 2019 (FTP do IBGE): https://ftp.ibge.gov.br/PNS/2019/Microdados/
- Portal da PNS (Fiocruz/ICICT), com bases e dicionários: https://www.pns.icict.fiocruz.br/bases-de-dados/

O dicionário de variáveis usado foi `dicionario_PNS_microdados_2019_23062023.xlsx`,
que vem junto com os microdados.

### Como citar (ABNT)

```
INSTITUTO BRASILEIRO DE GEOGRAFIA E ESTATÍSTICA (IBGE). Pesquisa Nacional de Saúde 2019:
percepção do estado de saúde, estilos de vida, doenças crônicas e saúde bucal: Brasil e
grandes regiões. Rio de Janeiro: IBGE, 2020.

INSTITUTO BRASILEIRO DE GEOGRAFIA E ESTATÍSTICA (IBGE). Pesquisa Nacional de Saúde – PNS 2019:
microdados. Rio de Janeiro: IBGE, 2020. Disponível em:
https://www.ibge.gov.br/estatisticas/sociais/saude/9160-pesquisa-nacional-de-saude.html.
```

## Recorte usado no projeto

Arquivo: [`../c45/dados_c45_sembalanceamento_originais.csv`](../c45/dados_c45_sembalanceamento_originais.csv)
(separador `;`, codificação UTF-8)

| | |
|---|---|
| Linhas | 53.630 |
| Colunas | 26 (25 atributos + alvo) |
| Alvo | `Q00201`: diagnóstico médico de hipertensão (`Sim` / `Nao`) |
| Distribuição do alvo | Nao: 33.057 (61,6%) · Sim: 20.573 (38,4%) |
| Valores ausentes | só em `M005010` (25.153) |

> O artigo cita **53.149 registros, 28 atributos e 58% / 42%**. A diferença vem das
> etapas do KNIME (remoção de linhas sem IMC, atributos derivados), aplicadas *depois*
> deste CSV.

### Atributos

| Código | Descrição (dicionário PNS 2019) |
|---|---|
| `A001` | Tipo do domicílio |
| `C001` | Número de pessoas no domicílio |
| `C006` | Sexo |
| `C008` | Idade do morador (discretizada em faixas) |
| `C009` | Cor ou raça |
| `C011` | Estado civil |
| `D001` | Sabe ler e escrever |
| `M005010` | Trabalha(va) habitualmente entre 20h e 5h (trabalho noturno) |
| `M01601` | Frequência com que se reuniu com outras pessoas para atividades esportivas, recreativas ou artísticas (12 meses) |
| `N010` | Problemas no sono (duas últimas semanas) |
| `N011` | Não se sentir descansado(a) / disposto(a) durante o dia |
| `N016` | Sentiu-se deprimido(a), "pra baixo" ou sem perspectiva |
| `P02601` | Autoavaliação do consumo de sal |
| `P034` | Praticou exercício físico ou esporte nos últimos três meses |
| `P038` | Anda bastante a pé no trabalho |
| `P039` | Faz atividade pesada / esforço físico intenso no trabalho |
| `P044` | Faz atividade pesada / esforço físico intenso nas atividades domésticas |
| `P046` | Existe lugar público perto do domicílio para caminhada/exercício |
| `Q00201` | **Alvo**: algum médico já lhe deu o diagnóstico de hipertensão arterial? |
| `VDD004A` | Nível de instrução mais elevado alcançado |
| `VDF004` | Faixa de rendimento domiciliar per capita |
| `VDM001` | Faixa de tempo gasto por dia no deslocamento casa-trabalho |
| `Status_N017` | Derivado de `N017`: sentiu-se mal consigo mesmo / um fracasso |
| `Status_P02002` | Derivado de `P02002`: dias por semana que toma refrigerante |
| `Status_P02602` | Derivado de `P02602`: dias por semana que substitui o almoço por lanches |
| `Status_P02801` | Derivado de `P02801`: dias por semana que consome bebida alcoólica |

> Descrições resumidas do dicionário oficial da PNS 2019. As colunas `Status_*` são
> versões discretizadas/combinadas das variáveis originais, criadas no pré-processamento.
