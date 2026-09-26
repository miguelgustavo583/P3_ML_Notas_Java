/**
 * Regressão Linear Múltipla resolvida pela Equação Normal:
 *   theta = (X^T X)^-1 X^T y
 * X já deve incluir a coluna de 1's (intercepto) como primeira coluna.
 */
public class LinearRegressionOLS {

    public double[] theta; // theta[0] = intercepto, theta[1..n] = coeficientes

    public void fit(double[][] X, double[] y) {
        double[][] Xt = Matrix.transpose(X);
        double[][] XtX = Matrix.multiply(Xt, X);
        double[][] XtXinv = Matrix.inverse(XtX);
        double[] Xty = Matrix.multiply(Xt, y);
        this.theta = Matrix.multiply(XtXinv, Xty);
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
