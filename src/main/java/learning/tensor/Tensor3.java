package learning.tensor;

public class Tensor3 extends Tensor {
   
   private float[][][] v;
   
   public Tensor3(int[] s) {
      super(s);
      v = new float[size[0]][size[1]][size[2]];
   }
   
   public float[] getVector(int i) {
      int x = (int) (i/size[1]);
      int y = i % size[1];
      return v[x][y];
   }
   
   public int vectorNum() {
      return size[0]*size[1];
   }
   
   public float[][] getMatrix(int i) {
      return v[i];
   }
   
   public int matrixNum() {
      return size[0];
   }
   
   public void load(float[][][] d) {
      v = d;
   }
   
}
