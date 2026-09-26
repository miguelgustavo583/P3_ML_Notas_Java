import java.util.TreeMap;

public class StatsUtil {

    public static double mean(double[] v) {
        double s = 0;
        for (double x : v) s += x;
        return s / v.length;
    }

    /** Desvio padrão amostral (ddof = 1), igual ao numpy.std(ddof=1). */
    public static double stdDev(double[] v) {
        double m = mean(v);
        double s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return Math.sqrt(s / (v.length - 1));
    }

    public static double mae(double[] real, double[] previsto) {
        double s = 0;
        for (int i = 0; i < real.length; i++) s += Math.abs(real[i] - previsto[i]);
        return s / real.length;
    }

    public static double rmse(double[] real, double[] previsto) {
        double s = 0;
        for (int i = 0; i < real.length; i++) s += Math.pow(real[i] - previsto[i], 2);
        return Math.sqrt(s / real.length);
    }

    public static double r2(double[] real, double[] previsto) {
        double media = mean(real);
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < real.length; i++) {
            ssRes += Math.pow(real[i] - previsto[i], 2);
            ssTot += Math.pow(real[i] - media, 2);
        }
        return 1.0 - (ssRes / ssTot);
    }

    public static double correlacao(double[] x, double[] y) {
        double mx = mean(x), my = mean(y);
        double num = 0, dx = 0, dy = 0;
        for (int i = 0; i < x.length; i++) {
            num += (x[i] - mx) * (y[i] - my);
            dx += Math.pow(x[i] - mx, 2);
            dy += Math.pow(y[i] - my, 2);
        }
        return num / Math.sqrt(dx * dy);
    }

    /**
     * Valor crítico t de Student (bicaudal, 95%) por interpolação de tabela.
     * Evita depender de bibliotecas externas de estatística.
     */
    private static final TreeMap<Integer, Double> T_TABLE = new TreeMap<>();
    static {
        T_TABLE.put(1, 12.706); T_TABLE.put(2, 4.303); T_TABLE.put(3, 3.182);
        T_TABLE.put(4, 2.776);  T_TABLE.put(5, 2.571); T_TABLE.put(6, 2.447);
        T_TABLE.put(7, 2.365);  T_TABLE.put(8, 2.306); T_TABLE.put(9, 2.262);
        T_TABLE.put(10, 2.228); T_TABLE.put(15, 2.131); T_TABLE.put(20, 2.086);
        T_TABLE.put(25, 2.060); T_TABLE.put(30, 2.042); T_TABLE.put(40, 2.021);
        T_TABLE.put(60, 2.000); T_TABLE.put(120, 1.980); T_TABLE.put(100000, 1.960);
    }

    public static double tCritico95(int df) {
        if (T_TABLE.containsKey(df)) return T_TABLE.get(df);
        Integer dfBaixo = T_TABLE.floorKey(df);
        Integer dfAlto = T_TABLE.ceilingKey(df);
        if (dfBaixo == null) return T_TABLE.firstEntry().getValue();
        if (dfAlto == null) return T_TABLE.lastEntry().getValue();
        double tBaixo = T_TABLE.get(dfBaixo), tAlto = T_TABLE.get(dfAlto);
        double frac = (double) (df - dfBaixo) / (dfAlto - dfBaixo);
        return tBaixo + frac * (tAlto - tBaixo);
    }
}
