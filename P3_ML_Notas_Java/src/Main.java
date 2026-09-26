import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * P3 - Machine Learning para previsão de nota final de alunos (versão Java)
 * Integrantes: Miguel Gustavo de Sousa Campos e Henrique de Moraes Rodrigues
 *
 * Etapas:
 * 1. Geração de uma base fake de alunos (horas de estudo, frequência,
 *    atividades entregues, nota anterior, nota final)
 * 2. Treinamento de dois modelos de Regressão Linear:
 *      - OLS (Equação Normal, solução fechada)
 *      - Gradiente Descendente (iterativo), para comparação
 * 3. Cálculo de métricas estatísticas (erro médio, desvio padrão,
 *    MAE, RMSE, R², intervalo de erro / IC 95%)
 * 4. Geração de gráficos (PNG, via java.awt.Graphics2D)
 * 5. Impressão dos resultados para interpretação
 */
public class Main {

    static final int N = 300;
    static final long SEED = 42L;
    static final String OUT_DIR = "saida/";

    public static void main(String[] args) throws IOException {
        new File(OUT_DIR).mkdirs();
        Random rnd = new Random(SEED);

        // -----------------------------------------------------------
        // 1. GERAÇÃO DA BASE FAKE DE ALUNOS
        // -----------------------------------------------------------
        double[] horasEstudo = new double[N];
        double[] frequencia = new double[N];
        double[] atividades = new double[N];
        double[] notaAnterior = new double[N];
        double[] notaFinal = new double[N];

        for (int i = 0; i < N; i++) {
            horasEstudo[i] = clip(gauss(rnd, 8.0, 3.5), 0, 20);
            frequencia[i] = clip(gauss(rnd, 85.0, 10.0), 40, 100);
            atividades[i] = Math.round(clip(gauss(rnd, 8.0, 2.0), 0, 10));
            notaAnterior[i] = clip(gauss(rnd, 6.5, 1.5), 0, 10);

            double ruido = gauss(rnd, 0.0, 0.6);
            double nf = 0.28 * horasEstudo[i]
                      + 0.045 * frequencia[i]
                      + 0.35 * atividades[i]
                      + 0.35 * notaAnterior[i]
                      - 4.0
                      + ruido;
            notaFinal[i] = clip(nf, 0, 10);
        }

        salvarCsv(horasEstudo, frequencia, atividades, notaAnterior, notaFinal, OUT_DIR + "base_alunos_fake.csv");
        System.out.println("Base gerada com " + N + " alunos -> " + OUT_DIR + "base_alunos_fake.csv");

        // -----------------------------------------------------------
        // 2. SPLIT TREINO/TESTE (75% / 25%)
        // -----------------------------------------------------------
        Integer[] idx = new Integer[N];
        for (int i = 0; i < N; i++) idx[i] = i;
        List<Integer> lista = new ArrayList<>(Arrays.asList(idx));
        Collections.shuffle(lista, new Random(SEED));
        int nTeste = (int) Math.round(N * 0.25);
        List<Integer> testeIdx = lista.subList(0, nTeste);
        List<Integer> treinoIdx = lista.subList(nTeste, N);

        double[][] XtreinoRaw = montarX(treinoIdx, horasEstudo, frequencia, atividades, notaAnterior);
        double[][] XtesteRaw = montarX(testeIdx, horasEstudo, frequencia, atividades, notaAnterior);
        double[] yTreino = extrair(treinoIdx, notaFinal);
        double[] yTeste = extrair(testeIdx, notaFinal);

        // -----------------------------------------------------------
        // 3. MODELO PRINCIPAL: REGRESSÃO LINEAR (EQUAÇÃO NORMAL / OLS)
        // -----------------------------------------------------------
        double[][] XtreinoOLS = comIntercepto(XtreinoRaw);
        double[][] XtesteOLS = comIntercepto(XtesteRaw);

        LinearRegressionOLS ols = new LinearRegressionOLS();
        ols.fit(XtreinoOLS, yTreino);
        double[] predOLS = ols.predict(XtesteOLS);

        double[] erros = new double[yTeste.length];
        for (int i = 0; i < yTeste.length; i++) erros[i] = yTeste[i] - predOLS[i];

        double mediaErro = StatsUtil.mean(erros);
        double stdErro = StatsUtil.stdDev(erros);
        double mae = StatsUtil.mae(yTeste, predOLS);
        double rmse = StatsUtil.rmse(yTeste, predOLS);
        double r2 = StatsUtil.r2(yTeste, predOLS);

        int df = yTeste.length - 1;
        double tCrit = StatsUtil.tCritico95(df);
        double margem = tCrit * (stdErro / Math.sqrt(yTeste.length));
        double icInf = mediaErro - margem;
        double icSup = mediaErro + margem;

        System.out.println("\n===== METRICAS - REGRESSAO LINEAR (OLS / Equacao Normal) =====");
        System.out.printf("Erro medio: %.4f%n", mediaErro);
        System.out.printf("Desvio padrao do erro: %.4f%n", stdErro);
        System.out.printf("MAE: %.4f%n", mae);
        System.out.printf("RMSE: %.4f%n", rmse);
        System.out.printf("R2: %.4f%n", r2);
        System.out.printf("Intervalo de erro (IC 95%%, t critico=%.3f, df=%d): [%.4f, %.4f]%n", tCrit, df, icInf, icSup);
        System.out.println("Coeficientes: intercepto=" + fmt(ols.theta[0])
                + ", horas_estudo=" + fmt(ols.theta[1])
                + ", frequencia=" + fmt(ols.theta[2])
                + ", atividades=" + fmt(ols.theta[3])
                + ", nota_anterior=" + fmt(ols.theta[4]));

        // -----------------------------------------------------------
        // 4. MODELO DE COMPARAÇÃO: REGRESSÃO LINEAR VIA GRADIENTE DESCENDENTE
        //    (features padronizadas em z-score para estabilidade)
        // -----------------------------------------------------------
        double[] mediaFeat = new double[4], stdFeat = new double[4];
        for (int j = 0; j < 4; j++) {
            double[] col = coluna(XtreinoRaw, j);
            mediaFeat[j] = StatsUtil.mean(col);
            stdFeat[j] = StatsUtil.stdDev(col);
        }
        double[][] XtreinoStd = padronizarComIntercepto(XtreinoRaw, mediaFeat, stdFeat);
        double[][] XtesteStd = padronizarComIntercepto(XtesteRaw, mediaFeat, stdFeat);

        LinearRegressionGD gd = new LinearRegressionGD();
        gd.fit(XtreinoStd, yTreino, 0.05, 3000);
        double[] predGD = gd.predict(XtesteStd);

        double maeGD = StatsUtil.mae(yTeste, predGD);
        double rmseGD = StatsUtil.rmse(yTeste, predGD);
        double r2GD = StatsUtil.r2(yTeste, predGD);

        System.out.println("\n===== METRICAS - REGRESSAO LINEAR (Gradiente Descendente, comparacao) =====");
        System.out.printf("MAE: %.4f | RMSE: %.4f | R²: %.4f%n", maeGD, rmseGD, r2GD);

        // -----------------------------------------------------------
        // 5. GRÁFICOS
        // -----------------------------------------------------------
        // Amostra de 20 alunos do teste, espaçados uniformemente pela nota real,
        // para o gráfico de colunas (real x prevista) ficar legível.
        int amostraN = Math.min(20, yTeste.length);
        Integer[] ordemPorNota = new Integer[yTeste.length];
        for (int i = 0; i < yTeste.length; i++) ordemPorNota[i] = i;
        Arrays.sort(ordemPorNota, Comparator.comparingDouble(i -> yTeste[i]));
        double[] realAmostra = new double[amostraN];
        double[] prevAmostra = new double[amostraN];
        String[] labelsAmostra = new String[amostraN];
        for (int k = 0; k < amostraN; k++) {
            int pos = (int) Math.round(k * (yTeste.length - 1.0) / (amostraN - 1.0));
            int i = ordemPorNota[pos];
            realAmostra[k] = yTeste[i];
            prevAmostra[k] = predOLS[i];
            labelsAmostra[k] = "A" + (k + 1);
        }
        ChartUtil.colunasRealVsPrevisto(labelsAmostra, realAmostra, prevAmostra, OUT_DIR + "grafico_real_vs_previsto.png");
        ChartUtil.histogramaErros(erros, mediaErro, OUT_DIR + "grafico_distribuicao_erros.png");

        // Coeficientes padronizados para o gráfico de importância (usa o modelo OLS treinado com z-score)
        LinearRegressionOLS olsStd = new LinearRegressionOLS();
        olsStd.fit(XtreinoStd, yTreino);
        String[] labels = {"nota_anterior", "atividades_entregues", "frequencia_pct", "horas_estudo"};
        double[] valoresImportancia = {
                Math.abs(olsStd.theta[4]), Math.abs(olsStd.theta[3]),
                Math.abs(olsStd.theta[2]), Math.abs(olsStd.theta[1])
        };
        ChartUtil.barrasImportancia(labels, valoresImportancia, OUT_DIR + "grafico_importancia_variaveis.png");

        String[] labelsCorr = {"horas_estudo", "frequencia_pct", "atividades", "nota_anterior", "nota_final"};
        double[][] todasColunas = {horasEstudo, frequencia, atividades, notaAnterior, notaFinal};
        double[][] corr = new double[5][5];
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                corr[i][j] = (i == j) ? 1.0 : StatsUtil.correlacao(todasColunas[i], todasColunas[j]);
        ChartUtil.heatmapCorrelacao(labelsCorr, corr, OUT_DIR + "grafico_correlacao.png");

        // -----------------------------------------------------------
        // 6. SALVAR RESUMO EM TEXTO
        // -----------------------------------------------------------
        StringBuilder sb = new StringBuilder();
        sb.append("METRICAS - REGRESSAO LINEAR OLS (modelo principal)\n");
        sb.append(String.format("Erro medio: %.4f%n", mediaErro));
        sb.append(String.format("Desvio padrao do erro: %.4f%n", stdErro));
        sb.append(String.format("MAE: %.4f%n", mae));
        sb.append(String.format("RMSE: %.4f%n", rmse));
        sb.append(String.format("R2: %.4f%n", r2));
        sb.append(String.format("Intervalo de erro (IC 95%%): [%.4f, %.4f]%n%n", icInf, icSup));
        sb.append("METRICAS - REGRESSAO LINEAR GD (comparacao)\n");
        sb.append(String.format("MAE: %.4f%nRMSE: %.4f%nR2: %.4f%n", maeGD, rmseGD, r2GD));
        Files.writeString(Paths.get(OUT_DIR + "resultados.txt"), sb.toString());

        System.out.println("\nArquivos e graficos salvos em: " + new File(OUT_DIR).getAbsolutePath());
    }

    // ---------------- Auxiliares ----------------

    static double gauss(Random rnd, double media, double desvio) {
        return media + desvio * rnd.nextGaussian();
    }

    static double clip(double v, double min, double max) {
        return Math.max(min, Math.min(max, v));
    }

    static String fmt(double v) {
        return String.format(Locale.US, "%.4f", v);
    }

    static double[][] montarX(List<Integer> idxs, double[] a, double[] b, double[] c, double[] d) {
        double[][] X = new double[idxs.size()][4];
        for (int i = 0; i < idxs.size(); i++) {
            int k = idxs.get(i);
            X[i][0] = a[k]; X[i][1] = b[k]; X[i][2] = c[k]; X[i][3] = d[k];
        }
        return X;
    }

    static double[] extrair(List<Integer> idxs, double[] v) {
        double[] r = new double[idxs.size()];
        for (int i = 0; i < idxs.size(); i++) r[i] = v[idxs.get(i)];
        return r;
    }

    static double[][] comIntercepto(double[][] X) {
        double[][] r = new double[X.length][X[0].length + 1];
        for (int i = 0; i < X.length; i++) {
            r[i][0] = 1.0;
            System.arraycopy(X[i], 0, r[i], 1, X[i].length);
        }
        return r;
    }

    static double[] coluna(double[][] X, int j) {
        double[] c = new double[X.length];
        for (int i = 0; i < X.length; i++) c[i] = X[i][j];
        return c;
    }

    static double[][] padronizarComIntercepto(double[][] X, double[] media, double[] std) {
        double[][] r = new double[X.length][X[0].length + 1];
        for (int i = 0; i < X.length; i++) {
            r[i][0] = 1.0;
            for (int j = 0; j < X[0].length; j++) r[i][j + 1] = (X[i][j] - media[j]) / std[j];
        }
        return r;
    }

    static void salvarCsv(double[] h, double[] f, double[] a, double[] n, double[] nf, String path) throws IOException {
        StringBuilder sb = new StringBuilder("horas_estudo,frequencia_pct,atividades_entregues,nota_anterior,nota_final\n");
        for (int i = 0; i < h.length; i++) {
            sb.append(String.format(Locale.US, "%.1f,%.1f,%d,%.1f,%.1f%n",
                    h[i], f[i], (int) a[i], n[i], nf[i]));
        }
        Files.writeString(Paths.get(path), sb.toString());
    }
}
