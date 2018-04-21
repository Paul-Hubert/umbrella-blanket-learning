package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class RecurrentGate extends Input {
   
   private int time = 0;
   
   public RecurrentGate (int[] size) {
      super(size);
      outputSize = size;
   }
   
   @Override
   public void forwardProp() {
      time++;
   }
   
   @Override
   public void backwardProp() {
      time--;
      input = dO;
   }
   
   @Override
   Tensor forwards() {
      net.operate(this);
      return output;
   }
   
   @Override
   void backwards(Tensor t) {
      dO = t;
      if(time > 0) {
         net.operate(this);
         before.backwards(input);
      }
   }
   
   @Override
   protected void attachBefore(Node n) {
      before = n;
      if(outputSize[0] != before.outputSize[0]) {
         System.err.println("RecurrentGate different input/output sizes");
      }
   }
   
   @Override
   public void load(Tensor h0) {
      output = h0;
   }
   
}