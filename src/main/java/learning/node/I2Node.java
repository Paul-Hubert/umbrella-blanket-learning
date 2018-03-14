package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class I2Node extends Node {
   
   protected Node before2;
   
   protected Tensor input2, in2;
   
   void forwardProp() {
      output = in;
   }
   
   void backwardProp() {
      input = dO;
      input2 = dO;
   }
   
   Tensor forwards(Operation op) {
      in = before.forwards(op);
      in2 = before2.forwards(op);
      if(op.calculate) forwardProp();
      op.operate(this,output);
      return output;
   }
   
   void backwards(Tensor t, Operation op) {
      dO = t;
      if(op.calculate) backwardProp();
      op.operate(this,input);
      op.operate(this,input2);
      before.backwards(input, op);
      before2.backwards(input2, op);
   }
   
   protected void attachBefore(Node n) {
      if(before == null) {
         before = n;
         return;
      }
      before2 = n;
      
      setOutputSize();
      prepare();
   }
}