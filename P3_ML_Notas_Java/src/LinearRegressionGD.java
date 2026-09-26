/**
 * Regressão Linear treinada iterativamente por Gradiente Descendente,
 * usada como segundo modelo de Machine Learning para comparação com a
 * solução fechada (Equação Normal) do LinearRegressionOLS.
 * As features de entrada devem estar padronizadas (z-score) para a
 * convergência ser estável.
 */
public class LinearRegressionGD {

    public double[] theta;
    public double[] historicoCusto;

    public void fit(double[][] X, double[] y, double taxaAprendizado, int epocas) {
        int m = X.length, n = X[0].length;
        theta = new double[n];
        historicoCusto = new double[epocas];

        for (int epoca = 0; epoca < epocas; epoca++) {
            double[] pred = predict(X);
            double[] gradiente = new double[n];
            double custo = 0;

            for (int i = 0; i < m; i++) {
                double erro = pred[i] - y[i];
                custo += erro * erro;
                for (int j = 0; j < n; j++) gradiente[j] += erro * X[i][j];
            }
            for (int j = 0; j < n; j++) theta[j] -= taxaAprendizado * (gradiente[j] / m);
            historicoCusto[epoca] = custo / (2.0 * m);
        }
    }

    public double[] predict(double[][] X) {
        double[] pred = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double s = 0;
            for (int j = 0; j < theta.length; j++) s += X[i][j] * theta[j];
            pred[i] = s;
        }
        return pred;
    }
}
