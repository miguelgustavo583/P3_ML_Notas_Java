/**
 * Utilitário simples de álgebra matricial (sem bibliotecas externas),
 * usado para resolver a Regressão Linear pela Equação Normal:
 *   theta = (X^T * X)^-1 * X^T * y
 */
public class Matrix {

    public static double[][] transpose(double[][] a) {
        int rows = a.length, cols = a[0].length;
        double[][] t = new double[cols][rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j][i] = a[i][j];
        return t;
    }

    public static double[][] multiply(double[][] a, double[][] b) {
        int rows = a.length, inner = b.length, cols = b[0].length;
        double[][] r = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                for (int k = 0; k < inner; k++) sum += a[i][k] * b[k][j];
                r[i][j] = sum;
            }
        return r;
    }

    public static double[] multiply(double[][] a, double[] v) {
        int rows = a.length, cols = a[0].length;
        double[] r = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) sum += a[i][j] * v[j];
            r[i] = sum;
        }
        return r;
    }

    /** Inversão por eliminação de Gauss-Jordan com pivotamento parcial. */
    public static double[][] inverse(double[][] a) {
        int n = a.length;
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, aug[i], 0, n);
            aug[i][n + i] = 1.0;
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
            double[] tmp = aug[col]; aug[col] = aug[pivot]; aug[pivot] = tmp;

            double pivotVal = aug[col][col];
            if (Math.abs(pivotVal) < 1e-12)
                throw new ArithmeticException("Matriz singular - não é possível inverter.");
            for (int j = 0; j < 2 * n; j++) aug[col][j] /= pivotVal;

            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = aug[row][col];
                for (int j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++)
            System.arraycopy(aug[i], n, inv[i], 0, n);
        return inv;
    }
}
