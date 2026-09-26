import java.awt.*;
import java.awt.geom.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Utilitário de gráficos usando apenas java.awt.Graphics2D + ImageIO
 * (sem bibliotecas externas como JFreeChart), gerando PNGs.
 */
public class ChartUtil {

    static final Color NAVY = new Color(0x1F, 0x4E, 0x79);
    static final Color BLUE = new Color(0x4F, 0x81, 0xBD);
    static final Color ORANGE = new Color(0xED, 0x7D, 0x31);
    static final Font FONT_TITLE = new Font("SansSerif", Font.BOLD, 18);
    static final Font FONT_LABEL = new Font("SansSerif", Font.PLAIN, 14);

    private static Graphics2D setup(BufferedImage img) {
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, img.getWidth(), img.getHeight());
        return g;
    }

    /**
     * Gráfico de colunas (barras agrupadas) comparando nota real x nota
     * prevista para uma amostra de alunos do conjunto de teste.
     */
    public static void colunasRealVsPrevisto(String[] labels, double[] real, double[] previsto, String path) throws IOException {
        int n = labels.length;
        int margin = 70, top = 90, bottom = 70;
        int groupW = Math.max(28, Math.min(46, 900 / n));
        int barW = groupW / 2 - 2;
        int W = margin * 2 + n * groupW + 40;
        int H = 480;
        int plotH = H - top - bottom;

        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = setup(img);

        g.setFont(FONT_TITLE);
        g.setColor(NAVY);
        drawCentered(g, "Nota real vs. Nota prevista (Regressão Linear - OLS)", W / 2, 28);

        double max = 10.0;
        for (double v : real) max = Math.max(max, v);
        for (double v : previsto) max = Math.max(max, v);
        max = Math.ceil(max);

        // linhas de grade horizontais
        g.setFont(FONT_LABEL);
        int passo = max > 10 ? 2 : 2;
        for (int v = 0; v <= max; v += passo) {
            int y = H - bottom - (int) (v / max * plotH);
            g.setColor(new Color(230, 230, 230));
            g.drawLine(margin - 10, y, W - margin + 10, y);
            g.setColor(Color.BLACK);
            g.drawString(String.valueOf(v), margin - 30, y + 5);
        }
        g.setColor(Color.GRAY);
        g.drawLine(margin - 10, H - bottom, W - margin + 10, H - bottom);

        for (int i = 0; i < n; i++) {
            int gx = margin + i * groupW;
            int hReal = (int) (real[i] / max * plotH);
            int hPrev = (int) (previsto[i] / max * plotH);

            g.setColor(NAVY);
            g.fillRect(gx, H - bottom - hReal, barW, hReal);
            g.setColor(ORANGE);
            g.fillRect(gx + barW + 3, H - bottom - hPrev, barW, hPrev);

            g.setColor(Color.BLACK);
            Font old = g.getFont();
            g.setFont(new Font("SansSerif", Font.PLAIN, 10));
            String lbl = labels[i];
            FontMetrics fm = g.getFontMetrics();
            g.drawString(lbl, gx + groupW / 2 - fm.stringWidth(lbl) / 2, H - bottom + 16);
            g.setFont(old);
        }

        // legenda
        int lx = W / 2 - 110, ly = H - 28;
        g.setColor(NAVY);
        g.fillRect(lx, ly, 14, 14);
        g.setColor(Color.BLACK);
        g.setFont(FONT_LABEL);
        g.drawString("Nota real", lx + 20, ly + 12);
        g.setColor(ORANGE);
        g.fillRect(lx + 130, ly, 14, 14);
        g.setColor(Color.BLACK);
        g.drawString("Nota prevista", lx + 150, ly + 12);

        g.dispose();
        ImageIO.write(img, "png", new File(path));
    }

    public static void scatterRealVsPrevisto(double[] real, double[] previsto, String path) throws IOException {
        int W = 700, H = 700, margin = 80;
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = setup(img);

        g.setFont(FONT_TITLE);
        g.setColor(NAVY);
        drawCentered(g, "Nota real vs. Nota prevista (Regressão Linear - OLS)", W / 2, 30);

        double min = 0, max = 10;
        int plotSize = W - 2 * margin;

        g.setColor(Color.LIGHT_GRAY);
        g.setStroke(new BasicStroke(1));
        for (int i = 0; i <= 10; i += 2) {
            int x = margin + (int) ((i - min) / (max - min) * plotSize);
            int y = H - margin - (int) ((i - min) / (max - min) * plotSize);
            g.drawLine(x, margin, x, H - margin);
            g.drawLine(margin, y, W - margin, y);
        }

        g.setColor(Color.GRAY);
        g.drawRect(margin, margin, plotSize, plotSize);
        float[] dash = {6f, 6f};
        g.setStroke(new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, dash, 0f));
        g.drawLine(margin, H - margin, W - margin, margin);

        g.setColor(BLUE);
        g.setStroke(new BasicStroke(1.5f));
        for (int i = 0; i < real.length; i++) {
            int x = margin + (int) ((real[i] - min) / (max - min) * plotSize);
            int y = H - margin - (int) ((previsto[i] - min) / (max - min) * plotSize);
            x = Math.max(margin, Math.min(W - margin, x));
            y = Math.max(margin, Math.min(H - margin, y));
            g.fillOval(x - 4, y - 4, 8, 8);
        }

        g.setFont(FONT_LABEL);
        g.setColor(Color.BLACK);
        for (int i = 0; i <= 10; i += 2) {
            int x = margin + (int) ((i - min) / (max - min) * plotSize);
            int y = H - margin - (int) ((i - min) / (max - min) * plotSize);
            g.drawString(String.valueOf(i), x - 5, H - margin + 20);
            g.drawString(String.valueOf(i), margin - 25, y + 5);
        }
        drawCentered(g, "Nota real", W / 2, H - 25);
        drawVertical(g, "Nota prevista", 25, H / 2);

        g.dispose();
        ImageIO.write(img, "png", new File(path));
    }

    public static void histogramaErros(double[] erros, double media, String path) throws IOException {
        int W = 700, H = 500, margin = 80;
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = setup(img);

        g.setFont(FONT_TITLE);
        g.setColor(NAVY);
        drawCentered(g, "Distribuição dos erros de previsão", W / 2, 30);

        int nBins = 14;
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double e : erros) { min = Math.min(min, e); max = Math.max(max, e); }
        double range = max - min;
        int[] bins = new int[nBins];
        for (double e : erros) {
            int idx = (int) ((e - min) / range * (nBins - 1e-9));
            if (idx >= nBins) idx = nBins - 1;
            if (idx < 0) idx = 0;
            bins[idx]++;
        }
        int maxCount = 0;
        for (int c : bins) maxCount = Math.max(maxCount, c);

        int plotW = W - 2 * margin, plotH = H - 2 * margin;
        double binW = (double) plotW / nBins;

        g.setColor(BLUE);
        for (int i = 0; i < nBins; i++) {
            int barH = (int) ((double) bins[i] / maxCount * plotH);
            int x = (int) (margin + i * binW);
            int y = H - margin - barH;
            g.fillRect(x + 1, y, (int) binW - 2, barH);
        }
        g.setColor(Color.GRAY);
        g.drawLine(margin, H - margin, W - margin, H - margin);
        g.drawLine(margin, margin, margin, H - margin);

        int xMedia = (int) (margin + (media - min) / range * plotW);
        g.setColor(Color.RED);
        float[] dash = {6f, 6f};
        g.setStroke(new BasicStroke(1.8f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, dash, 0f));
        g.drawLine(xMedia, margin, xMedia, H - margin);

        g.setFont(FONT_LABEL);
        g.setColor(Color.BLACK);
        drawCentered(g, String.format("Erro (real - previsto)  |  média = %.2f", media), W / 2, H - 25);

        g.dispose();
        ImageIO.write(img, "png", new File(path));
    }

    public static void barrasImportancia(String[] labels, double[] valores, String path) throws IOException {
        int W = 700, H = 420, margin = 140;
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = setup(img);

        g.setFont(FONT_TITLE);
        g.setColor(NAVY);
        drawCentered(g, "Importância relativa das variáveis (coef. padronizado)", W / 2, 30);

        double max = 0;
        for (double v : valores) max = Math.max(max, Math.abs(v));
        int plotW = W - margin - 40, plotH = H - 90;
        int barH = plotH / labels.length - 10;
        int x0 = margin;

        g.setFont(FONT_LABEL);
        for (int i = 0; i < labels.length; i++) {
            int y = 60 + i * (barH + 10);
            int w = (int) (valores[i] / max * plotW);
            g.setColor(BLUE);
            g.fillRect(x0, y, w, barH);
            g.setColor(Color.BLACK);
            g.drawString(labels[i], 10, y + barH / 2 + 5);
            g.drawString(String.format("%.3f", valores[i]), x0 + w + 8, y + barH / 2 + 5);
        }
        g.dispose();
        ImageIO.write(img, "png", new File(path));
    }

    public static void heatmapCorrelacao(String[] labels, double[][] corr, String path) throws IOException {
        int n = labels.length;
        int cell = 90, margin = 160, topGrid = 110;
        int W = margin + n * cell + 40, H = topGrid + n * cell + 20;
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = setup(img);

        g.setFont(FONT_TITLE);
        g.setColor(NAVY);
        drawCentered(g, "Matriz de correlação", W / 2, 25);

        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double v = corr[i][j];
                Color c = correlColor(v);
                int x = margin + j * cell, y = topGrid + i * cell;
                g.setColor(c);
                g.fillRect(x, y, cell, cell);
                g.setColor(Color.WHITE);
                g.drawRect(x, y, cell, cell);
                g.setColor(v > 0.6 || v < -0.6 ? Color.WHITE : Color.BLACK);
                String txt = String.format("%.2f", v);
                FontMetrics fm = g.getFontMetrics();
                g.drawString(txt, x + cell / 2 - fm.stringWidth(txt) / 2, y + cell / 2 + 5);
            }
        }
        g.setColor(Color.BLACK);
        for (int i = 0; i < n; i++) {
            g.drawString(labels[i], 10, topGrid + i * cell + cell / 2 + 5);
            AffineTransform old = g.getTransform();
            g.rotate(-Math.PI / 4, margin + i * cell + cell / 2, topGrid - 10);
            g.drawString(labels[i], margin + i * cell + 10, topGrid - 10);
            g.setTransform(old);
        }
        g.dispose();
        ImageIO.write(img, "png", new File(path));
    }

    private static Color correlColor(double v) {
        // Escala de azul: quanto maior a correlação, mais escuro.
        int intensity = (int) (255 - Math.abs(v) * 180);
        intensity = Math.max(40, Math.min(255, intensity));
        return new Color(intensity, intensity, 255);
    }

    private static void drawCentered(Graphics2D g, String text, int cx, int y) {
        FontMetrics fm = g.getFontMetrics();
        g.drawString(text, cx - fm.stringWidth(text) / 2, y);
    }

    private static void drawVertical(Graphics2D g, String text, int x, int cy) {
        AffineTransform old = g.getTransform();
        g.translate(x, cy);
        g.rotate(-Math.PI / 2);
        FontMetrics fm = g.getFontMetrics();
        g.drawString(text, -fm.stringWidth(text) / 2, 0);
        g.setTransform(old);
    }
}
