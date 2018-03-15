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
   void forwardProp() {
      time++;
   }
   
   @Override
   void backwardProp() {
      time--;
      input = dO;
   }
   
   @Override
   Tensor forwards(Operation op) {
      if(op.calculate) forwardProp();
      op.operate(this,output);
      return output;
   }
   
   @Override
   void backwards(Tensor t, Operation op) {
      dO = t;
      if(time > 0) {
         if(op.calculate) backwardProp();
         op.operate(this,input);
         before.backwards(input, op);
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