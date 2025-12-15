import hypermedia.net.*;

UDP udp;

float[] zHist;
float[] loadHist;
int histSize = 400;
int histPos = 0;

// 静的表示用のデフォルト範囲
float defaultZMin = -2;
float defaultZMax = 50;
// 力[N]として表示するためのデフォルト範囲
float defaultLoadMin = 0;
float defaultLoadMax = 9.81;  // ≒ 1.0 kg * 9.81

// 動的/静的の切り替えフラグ
boolean dynamicScale = true;

String lastZOp = "";
float latestZ = 0;
// latestLoad は「力[N]」を保持
float latestLoad = 0;

// 重力加速度 [m/s^2]（kg → N 変換用）
final float G = 9.80665;

void setup() {
  size(900, 500);
  udp = new UDP(this, 5005);
  udp.listen(true);

  zHist = new float[histSize];
  loadHist = new float[histSize];  // 中身は N 単位で格納
  for (int i = 0; i < histSize; i++) {
    zHist[i] = Float.NaN;
    loadHist[i] = Float.NaN;
  }

  textFont(createFont("Menlo", 12));
}

void receive(byte[] data, String ip, int port) {
  String s = new String(data).trim();
  // 例: "2025-11-06T00:12:34.123456,15.0,0.0,45.0,0.0997,y(+0.01)"
  String[] cols = splitTokens(s, ",");
  if (cols.length >= 5) {
    try {
      float z = float(cols[3]);
      // cols[4] は kg 単位の値として受け取り、N に変換
      float loadKg = float(cols[4]);
      float loadN = loadKg * G;

      latestZ = z;
      latestLoad = loadN;

      zHist[histPos] = z;
      loadHist[histPos] = loadN;  // N 単位で履歴に格納
      histPos = (histPos + 1) % histSize;

      if (cols.length >= 6) {
        lastZOp = cols[5];
      }
    } catch (Exception e) {
      println("parse error: " + e.getMessage());
    }
  }
}

void draw() {
  background(20);

  float leftMargin = 70;
  float rightMargin = 70;
  float topMargin = 60;
  float bottomMargin = 50;

  fill(255);
  textAlign(LEFT, TOP);
  text("Z / Load monitor", 10, 10);
  text("last Z op: " + (lastZOp.equals("") ? "(none)" : lastZOp), 10, 30);

  float gx1 = leftMargin;
  float gx2 = width - rightMargin;
  float gy1 = topMargin;
  float gy2 = height - bottomMargin;

  // 表示レンジを決定する
  float zMin, zMax, loadMin, loadMax;
  if (dynamicScale) {
    float[] zRange = getRangeFromHistory(zHist, defaultZMin, defaultZMax, 0.05);
    float[] loadRange = getRangeFromHistory(loadHist, defaultLoadMin, defaultLoadMax, 0.1);
    zMin = zRange[0];
    zMax = zRange[1];
    loadMin = loadRange[0];
    loadMax = loadRange[1];
  } else {
    // 静的表示
    zMin = defaultZMin;
    zMax = defaultZMax;
    loadMin = defaultLoadMin;
    loadMax = defaultLoadMax;
  }

  stroke(150);
  noFill();
  rect(gx1, gy1, gx2 - gx1, gy2 - gy1);

  drawLeftAxis(gx1, gy1, gy2, zMin, zMax);
  drawRightAxis(gx2, gy1, gy2, loadMin, loadMax);
  drawGraphs(gx1, gy1, gx2, gy2, zMin, zMax, loadMin, loadMax);

  // 最新値
  fill(255);
  textAlign(LEFT, BOTTOM);
  text(String.format("Z = %.3f mm", latestZ), 10, height - 30);
  text(String.format("Force = %.4f N", latestLoad), 10, height - 15);

  // 凡例（キーの説明＋線の色）
  drawKeyLegend(width - 210, 10);
}

// ====== 自動レンジ計算 ======
float[] getRangeFromHistory(float[] hist, float fallbackMin, float fallbackMax, float marginRate) {
  float vmin = Float.POSITIVE_INFINITY;
  float vmax = Float.NEGATIVE_INFINITY;
  int count = 0;
  for (int i = 0; i < hist.length; i++) {
    float v = hist[i];
    if (!Float.isNaN(v)) {
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
      count++;
    }
  }
  if (count <= 1) {
    return new float[]{fallbackMin, fallbackMax};
  }

  float range = vmax - vmin;
  if (range == 0) {
    vmin -= 0.5;
    vmax += 0.5;
  } else {
    float m = range * marginRate;
    vmin -= m;
    vmax += m;
  }

  return new float[]{vmin, vmax};
}

void drawLeftAxis(float x, float y1, float y2, float vmin, float vmax) {
  stroke(200);
  line(x, y1, x, y2);

  fill(200);
  textAlign(RIGHT, CENTER);

  int ticks = 6;
  for (int i = 0; i <= ticks; i++) {
    float t = map(i, 0, ticks, vmin, vmax);
    float yy = map(t, vmin, vmax, y2, y1);
    line(x - 5, yy, x, yy);
    text(nfc(t, 1), x - 7, yy);
  }
  textAlign(LEFT, TOP);
  text("Z (mm)", x + 3, y1 - 20);
}

void drawRightAxis(float x, float y1, float y2, float vmin, float vmax) {
  stroke(200);
  line(x, y1, x, y2);

  fill(200);
  textAlign(LEFT, CENTER);

  int ticks = 5;
  for (int i = 0; i <= ticks; i++) {
    float t = map(i, 0, ticks, vmin, vmax);
    float yy = map(t, vmin, vmax, y2, y1);
    line(x, yy, x + 5, yy);
    text(nfc(t, 2), x + 7, yy);   // N 表示なので小数2桁程度に
  }
  textAlign(RIGHT, TOP);
  text("Force (N)", x - 3, y1 - 20);
}

void drawGraphs(float gx1, float gy1, float gx2, float gy2,
                float zMin, float zMax, float loadMin, float loadMax) {

  // Zの線（水色）
  stroke(0, 200, 255);
  noFill();
  beginShape();
  for (int i = 0; i < histSize; i++) {
    int idx = (histPos + i) % histSize;
    float z = zHist[idx];
    if (Float.isNaN(z)) continue;
    float x = map(i, 0, histSize - 1, gx1, gx2);
    float y = map(z, zMin, zMax, gy2, gy1);
    vertex(x, y);
  }
  endShape();

  // Load（Force）の線（オレンジ）
  stroke(255, 180, 0);
  beginShape();
  for (int i = 0; i < histSize; i++) {
    int idx = (histPos + i) % histSize;
    float load = loadHist[idx];   // N 単位
    if (Float.isNaN(load)) continue;
    float x = map(i, 0, histSize - 1, gx1, gx2);
    float y = map(load, loadMin, loadMax, gy2, gy1);
    vertex(x, y);
  }
  endShape();
}


// 凡例とキー説明
void drawKeyLegend(float x, float y) {
  float w = 200;
  float h = 170;
  noStroke();
  fill(0, 150);     // 半透明の背景
  rect(x, y, w, h, 8);

  fill(255);
  textAlign(LEFT, TOP);
  text("Z-axis key map", x + 10, y + 8);

  int lineH = 18;
  float yy = y + 30;
  text("y / Y : Z ±0.01 mm", x + 10, yy); yy += lineH;
  text("u / U : Z ±0.001 mm", x + 10, yy); yy += lineH;
  text("t / T : Z ±0.1 mm", x + 10, yy); yy += lineH;
  text("m : toggle dynamic scale", x + 10, yy); yy += lineH + 6;

  // 線の凡例
  text("Legend:", x + 10, yy); yy += lineH;

  // Z line
  fill(0, 200, 255);
  rect(x + 10, yy + 4, 14, 4);
  fill(255);
  text("Z (mm)", x + 30, yy); yy += lineH;

  // Load(Force) line
  fill(255, 180, 0);
  rect(x + 10, yy + 4, 14, 4);
  fill(255);
  text("Force (N)", x + 30, yy);
}

// キー入力で動的/静的を切り替え
void keyPressed() {
  if (key == 'm' || key == 'M') {
    dynamicScale = !dynamicScale;
    println("dynamicScale = " + dynamicScale);
  }
}
