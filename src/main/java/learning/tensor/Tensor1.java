package learning.tensor;

class Tensor1 extends Tensor {
   
   private float[] v;
   
   protected Tensor1 (int[] s) {
      super(s);
      v = new float[size[0]];
   }
   
   public float[] getVector(int i) {
      return v;
   }
   
   public int vectorNum() {
      return 1;
   }
   
   public float[][] getMatrix(int i) {
      return null;
   }
   
   public int matrixNum() {
      return 0;
   }
   
   public void load(float[] d) {
      v = d;
   }
   
}