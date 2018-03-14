package learning.tensor;

class Tensor2 extends Tensor {
   
   private float[][] v;
   
   protected Tensor2 (int[] s) {
      super(s);
      v = new float[size[0]][size[1]];
   }
   
   public float[] getVector(int i) {
      return v[i];
   }
   
   public int vectorNum() {
      return size[0];
   }
   
   public float[][] getMatrix(int i) {
      return v;
   }
   
   public int matrixNum() {
      return 1;
   }
   
   public void load(float[][] d) {
      v = d;
   }
   
}