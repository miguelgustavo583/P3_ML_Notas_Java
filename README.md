# P3 - Machine Learning para Previsão de Nota Final de Alunos (Java)

Integrantes: Miguel Gustavo de Sousa Campos e Henrique de Moraes Rodrigues

100% Java puro (JDK), sem bibliotecas externas de ML, estatística ou gráficos:
- `Main.java` — orquestra geração de dados, treino e execução
- `Matrix.java` — álgebra matricial (multiplicação, transposta, inversão Gauss-Jordan)
- `LinearRegressionOLS.java` — Regressão Linear via Equação Normal (modelo principal)
- `LinearRegressionGD.java` — Regressão Linear via Gradiente Descendente (comparação)
- `StatsUtil.java` — média, desvio padrão, MAE, RMSE, R², correlação, t crítico (IC 95%)
- `ChartUtil.java` — gráficos PNG desenhados com java.awt.Graphics2D (sem JFreeChart)

## Como rodar
```bash
mkdir -p bin
javac -d bin src/*.java
java -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -cp bin Main
```
Gera a pasta `saida/` com o CSV, os 4 gráficos PNG e `resultados.txt`.

## Resumo dos resultados (Regressão Linear - OLS)
- R²: 0,8350
- MAE: 0,5441
- RMSE: 0,6780
- Erro médio: -0,0506 (desvio padrão 0,6807)
- Intervalo de erro (IC 95%): [-0,2074 ; 0,1063]
